import torch
import torch.nn as nn
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 获取GPU设备的数量
    num_gpu = torch.cuda.device_count()
    
    print("发现 {} 个可用的GPU:".format(num_gpu))
    
    for i in range(num_gpu):
        gpu_name = torch.cuda.get_device_name(i)
        print("GPU {}: {}".format(i, gpu_name))
else:
    print("没有可用的GPU，将在CPU上运行PyTorch。")


test_data = scipy.io.loadmat('sig_test_t.mat')
test_X = torch.tensor(test_data['sig_test'], dtype=torch.complex64)  # 用你的数据字段名称替换'test_X'
label=torch.tensor(test_data['chan']).int()
data_A=scipy.io.loadmat('A_t.mat')
A=torch.tensor(data_A['A'],dtype=torch.complex64)
# 创建测试数据集
test_dataset = TensorDataset(test_X,label)

# 创建测试数据的 DataLoader
batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#对应 t_cnn_bloss
class mCNN2(nn.Module):
    def __init__(self):
        super(mCNN2, self).__init__()
        self.conv1 = nn.Conv1d(2, 12, kernel_size=25, padding='same')
        self.conv2 = nn.Conv1d(12, 6, kernel_size=15, padding='same')
        self.conv3 = nn.Conv1d(6, 3, kernel_size=5, padding='same')
        self.conv4 = nn.Conv1d(3, 1, kernel_size=3, padding='same')
        self.activation = nn.Tanh()
        self.sigmoid=nn.Sigmoid()
 

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x

# 对应t_cnn.pth
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


# 加载已训练的模型
# load_path = 't_cnn.pth'  # 模型文件路径，请替换为你的实际路径
load_path = 't_cnn_bloss.pth'  # 模型文件路径，请替换为你的实际路径
checkpoint = torch.load(load_path)
# cnn_model = mCNN()
cnn_model = mCNN2()
cnn_model.load_state_dict(checkpoint['model_state'])

cnn_model.eval()
# criterion = nn.MSELoss()
criterion = nn.BCELoss()
# 迭代测试数据并进行推断
x=np.linspace(1,128,128*128)
for batch_X,batch_y in test_loader:
    # 进行推断
    batch_pro=torch.matmul(A,batch_X.t())
    batch_pro=batch_pro.t()
    batch_pro_r=torch.stack((batch_pro.real,batch_pro.imag),dim=1)#代理谱
    label_tensor=torch.zeros(batch_size, 1,128*128)#生成label
    for i in range(batch_size):
        tmp=batch_y[i]
        for j in range(4):
            if tmp[j*2]==128:
                break
            else:
                label_tensor[i,0,int(tmp[j*2]*128+tmp[j*2+1])]=1
    output = cnn_model(batch_pro_r)
    loss = criterion(output, label_tensor)
    plt.figure()
    output_detach = output.detach()
    output_np = output_detach.numpy()
    aaa=label_tensor.detach()
    label_tensor_np=aaa.numpy()
    plt.plot(x,label_tensor_np[0,0,:],label='true',color="r",linewidth=1)
    plt.plot(x,output_np[0,0,:],label='est',color="b",linewidth=2)
    plt.legend()
    plt.grid()
    plt.show()
    print(f"loss{loss.item()}")

# 输出模型的预测结果
print("模型的预测输出：")
print(output)