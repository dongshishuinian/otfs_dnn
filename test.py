import torch

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