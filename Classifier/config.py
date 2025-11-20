import torch

class Config:
    # 基础配置
    project_name = "CIFAR10_CLIP_FineTune"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    
    # 数据路径
    data_root = "./data"
    
    # 训练超参数
    # 注意：ViT-Large显存占用较大，如果显存不够(OOM)，请减小batch_size (如32) 并增加 accumulation_steps
    batch_size = 64 
    num_workers = 4
    epochs = 10  # CLIP微调收敛很快，10个epoch通常足够，想要极致99%可以设为20
    
    # 优化器参数
    learning_rate = 5e-6  # 微调需要极小的学习率
    weight_decay = 0.1
    
    # 模型参数
    model_name = "openai/clip-vit-large-patch14" # 使用OpenAI的CLIP Large模型
    num_classes = 10
    
    # 日志路径
    log_dir = "./Classifier"
    log_file = "Experiment1.log"