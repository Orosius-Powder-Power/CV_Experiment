import torch

class Config:
    project_name = "DeepLabV3Plus_VOC2012_SOTA"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "./data"
    
    # === SOTA 升级 1: 换成 ResNeSt-101 ===
    # 注意：需要 pip install timm
    # tu- 前缀表示使用 timm 库的实现，resnest101e 是加强版
    arch = "DeepLabV3Plus"
    encoder = "tu-resnest101e" 
    encoder_weights = "imagenet"
    
    num_classes = 21 
    
    # 显存压力会变大，如果 OOM，把 batch_size 降为 8
    # 但为了 SOTA，尽量用大 Batch (建议 16 或 8+GradientAccumulation)
    batch_size = 16 
    num_workers = 4 # 如果报错改回 2
    
    # ResNeSt 收敛需要细腻的调整，保持 0.007 或 0.01
    learning_rate = 0.007
    epochs = 150 # 150 轮足够榨干 ResNeSt
    
    crop_size = 512 
    save_dir = "./results"