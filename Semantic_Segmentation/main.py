import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os
import numpy as np

from config import Config
from dataset import VOCDataset, get_transforms

def calculate_miou(preds, targets, num_classes=21):
    """手动计算 mIoU"""
    # preds: [B, H, W], targets: [B, H, W]
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # 忽略 255 (边缘/背景)
    mask = (targets != 255)
    preds = preds[mask]
    targets = targets[mask]
    
    confusion_matrix = torch.bincount(
        num_classes * targets + preds, 
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    
    iou_per_class = torch.diag(confusion_matrix) / (
        confusion_matrix.sum(1) + confusion_matrix.sum(0) - torch.diag(confusion_matrix) + 1e-6
    )
    return iou_per_class.mean().item()

def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # 1. 数据
    train_tf, val_tf = get_transforms(cfg)
    
    train_ds = VOCDataset(root=cfg.data_root, year='2012', image_set='train', download=False, transform=train_tf)
    val_ds = VOCDataset(root=cfg.data_root, year='2012', image_set='val', download=False, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    # 2. 模型 (DeepLabV3+ with ResNet101)
    print(f"Building {cfg.arch} with {cfg.encoder}...")
    model = smp.DeepLabV3Plus(
        encoder_name=cfg.encoder, 
        encoder_weights=cfg.encoder_weights, 
        in_channels=3, 
        classes=cfg.num_classes
    )
    model.to(cfg.device)
    
    # 3. 优化器 & Loss
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    # 关键：忽略 255 标签 (VOC 的物体边缘)
    criterion = nn.CrossEntropyLoss(ignore_index=255) 
    
    # 4. 训练循环
    best_miou = 0.0
    
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
            
            optimizer.zero_grad()
            logits = model(imgs) # [B, 21, H, W]
            
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        # 验证
        model.eval()
        total_miou = 0
        count = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1) # [B, H, W]
                
                miou = calculate_miou(preds, masks, cfg.num_classes)
                total_miou += miou
                count += 1
        
        avg_miou = total_miou / count
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val mIoU: {avg_miou:.4f}")
        
        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_model.pth"))
            print(f"New Best mIoU: {best_miou:.4f} (Saved)")

    print(f"Final Best mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    main()