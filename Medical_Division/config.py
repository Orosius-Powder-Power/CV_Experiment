import torch

class Config:
    project_name = "ISBI_Segmentation_SOTA"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据路径 [cite: 625]
    train_img_path = "./data/isbi2012/train-volume.tif"
    train_mask_path = "./data/isbi2012/train-labels.tif"
    test_img_path = "./data/isbi2012/test-volume.tif"
    
    # SOTA 模型配置
    # 使用 U-Net++ (Nested U-Net) 配合 EfficientNet-b4 预训练编码器
    # 这比文档中的普通 U-Net 强很多
    arch = "UnetPlusPlus"
    encoder = "efficientnet-b4"
    encoder_weights = "imagenet"
    
    # 训练参数
    in_channels = 1  # 灰度图 
    classes = 1      # 二分类（前景/背景）
    
    # 由于只有30张图，我们需要较大的 Batch 和 激进的增强
    batch_size = 4   # 如果显存不够改为 2
    num_workers = 4
    learning_rate = 1e-4
    epochs = 50      # 30张图收敛很快，但需要多跑几轮配合增强
    
    # 图像尺寸，文档建议 Resize 到 572，U-Net++ 建议是 32 的倍数
    # 我们使用 512x512 (原图尺寸) 进行训练，避免 Resize 带来的插值损失
    img_size = 512 
    
    save_dir = "./results"