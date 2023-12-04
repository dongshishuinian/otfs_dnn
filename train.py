import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.init as init
import scipy.io

plt.ion()
train_losses = []
if torch.cuda.is_available():
    device=torch.device('cuda')
    print('Gpu可用')
else:
    print('Gpu不可用')

# 加载数据
data = scipy.io.loadmat('sig_gen_t.mat')
chan=data['chan']
sig=data['sig_train']
data_A=scipy.io.loadmat('A_t.mat')
A=data_A['A']

# 转换数据为 PyTorch 张量
label = torch.tensor(chan).int()
sig=torch.tensor(sig,dtype=torch.complex64)
A=torch.tensor(A,dtype=torch.complex64)
# 划分训练集和验证集
train_X, val_X, train_y, val_y = train_test_split(sig, label, test_size=0.2)

# 定义 CNN 模型
class mCNN(nn.Module):
    def __init__(self):
        super(mCNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 12, kernel_size=25, padding='same')
        self.conv2 = nn.Conv1d(12, 6, kernel_size=15, padding='same')
        self.conv3 = nn.Conv1d(6, 3, kernel_size=5, padding='same')
        self.conv4 = nn.Conv1d(3, 1, kernel_size=3, padding='same')
        self.activation = nn.Tanh()
 

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        return x

# 初始化模型和优化器
cnn_model = mCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# 训练模型
num_epochs = 600
batch_size = 8
train_loader = torch.utils.data.DataLoader(list(zip(train_X, train_y)), batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_size=batch_X.size(0)
        optimizer.zero_grad()
        batch_pro=torch.matmul(A,batch_X.t())
        batch_pro=batch_pro.t()
        batch_pro_r=torch.stack((batch_pro.real,batch_pro.imag),dim=1)#代理谱
        label_tensor=torch.zeros(batch_size, 128*128, 1)#生成label
        for i in range(batch_size):
            tmp=batch_y[i]
            label_tensor[i,int(tmp[0]*128+tmp[1]),0]=1
            label_tensor[i,int(tmp[2]*128+tmp[3]),0]=1
        output = cnn_model(batch_pro_r)
        loss = criterion(output, label_tensor)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"epoch:{epoch},loss:{avg_loss} \n")
    avg_loss=np.log(avg_loss)
    if epoch!=0:
        train_losses.append(avg_loss)
    
    # 清空当前轴并绘制损失曲线
    plt.clf()
    plt.plot(train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.pause(0.1)  # 暂停一小段时间以便刷新图像

#存
model_state = cnn_model.state_dict()
optimizer_state = optimizer.state_dict()
epoch = num_epochs  # 保存当前训练的轮数

# 指定保存模型的文件路径
save_path = 't_cnn_bloss.pth'  # 请替换为你的文件路径

# 创建一个字典来保存模型相关信息
model_info = {
    'epoch': epoch,
    'model_state': model_state,
    'optimizer_state': optimizer_state,
}

# 使用torch.save()保存模型信息到文件
torch.save(model_info, save_path)

# 关闭交互模式
plt.ioff()
plt.show()
# 验证模型
with torch.no_grad():
    val_output = cnn_model(val_X)
    val_loss = criterion(val_output, val_y)
    print(f"Validation Loss: {val_loss.item()}")

# 同样的方式，可以定义 DNN 模型并进行训练和验证
