import torch

class Config:
    project_name = "DeepLabV3Plus_VOC2012"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据路径 (torchvision下载后的默认路径)
    data_root = "./data"
    
    # SOTA 模型配置
    # 使用 DeepLabV3+ (比文档的 v3 更强)
    # Encoder 使用 ResNet-101 (深层网络提取特征更强)
    arch = "DeepLabV3Plus"
    encoder = "resnet101" 
    encoder_weights = "imagenet"
    
    # PASCAL VOC 类别 (20类 + 1背景)
    num_classes = 21 
    
    # 训练参数
    # 分割任务显存消耗大，ResNet101 + 512x512可能需要 8G+ 显存
    # 如果 OOM (Out of Memory)，请将 batch_size 调为 8 或 4
    batch_size = 16
    num_workers = 4
    learning_rate = 1e-4 # 微调学习率
    epochs = 200
    
    # 数据增强参数
    crop_size = 512 # 标准 VOC 输入尺寸
    
    # 结果保存
    save_dir = "./results"