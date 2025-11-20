import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPImageProcessor
from tqdm import tqdm
import os
import argparse

from config import Config
from model import CLIPClassifier
from utils import setup_logger, plot_training_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CIFAR-10 CLIP Fine-tuning')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cpu, cuda, cuda:0). If not specified, uses config default.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training. If not specified, uses config default.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs. If not specified, uses config default.')
    return parser.parse_args()

#HF_ENDPOINT=https://hf-mirror.com nohup python main.py --device cuda:3 > run.log 2>&1 &

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 1. 初始化配置和日志
    cfg = Config()
    
    # 覆盖配置中的参数（如果命令行提供了）
    if args.device is not None:
        cfg.device = args.device
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    
    logger = setup_logger(cfg.log_dir, cfg.log_file)
    logger.info(f"Project: {cfg.project_name}")
    logger.info(f"Using Device: {cfg.device}")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Epochs: {cfg.epochs}")

    # 2. 数据预处理
    # CLIP ViT需要特定的Normalization和Resize (224x224)
    processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
    
    # 使用processor自带的mean/std，确保与预训练一致
    # CIFAR10默认32x32，必须Resize到224才能利用ViT的Patch embedding
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(), # 简单增强
        transforms.RandomCrop(224, padding=4), # 防止过拟合
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    # 3. 加载CIFAR-10数据集
    # 文档提及类别: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck [cite: 114]
    logger.info("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=cfg.data_root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # 4. 模型构建
    model = CLIPClassifier(cfg.model_name, cfg.num_classes).to(cfg.device)

    # 5. 优化器与Scheduler
    # 使用AdamW，对于Transformer架构效果更好
    # AdamW(W 代表 Weight Decay decoupling):相比于Adam，对于权重衰退正则化项中从以动量模式的更新（即需要除以方差）中剥离出来，单独对这一部分进行标准的更新，从而使参数大的时候权重衰退量梯度下降地幅度也大，增加模型的泛化能力，适用于参数量大的Transformer架构中，防止过拟合。
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Cosine Annealing Scheduler
    # 学习率调度器，以余弦曲线让lr缓缓下降
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # 6. 训练循环
    best_acc = 0.0
    history = {'loss': [], 'train_acc': [], 'test_acc': []}

    logger.info("============== Starting Training ==============") 
    
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for images, labels in pbar:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({"Loss": loss.item()})
            
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # 评估测试集
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(cfg.device), labels.to(cfg.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        history['loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_acc'].append(test_acc)
        
        logger.info(f"Epoch: {epoch+1}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(cfg.log_dir, "best_model.pth"))
            logger.info(f"New best model saved with accuracy: {best_acc:.2f}%")

    logger.info(f"Final Best Test Accuracy: {best_acc:.2f}%")
    
    # 7. 绘图
    plot_save_path = os.path.join(cfg.log_dir, "training_curves.png")
    plot_training_results(history['loss'], history['train_acc'], history['test_acc'], plot_save_path)

if __name__ == '__main__':
    main()