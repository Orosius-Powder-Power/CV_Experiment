import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import segmentation_models_pytorch as smp
import os
import numpy as np

from config import Config
from dataset import ISBIDataset, get_transforms
from utils import dice_coef, plot_results

def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # 1. 数据准备
    train_tf, val_tf = get_transforms(cfg)
    full_dataset = ISBIDataset(cfg.train_img_path, cfg.train_mask_path, transform=train_tf)
    
    # 划分训练集和验证集 (90% 训练, 10% 验证) [cite: 930]
    n_val = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
    
    # 验证集使用 val_transform (这就需要稍微hack一下dataset，或者分两次实例化)
    # 简单起见，我们重新实例化验证集以应用正确的 transform
    # (严谨做法是 split indices 然后用 Subset 传不同的 transform)
    full_val_dataset = ISBIDataset(cfg.train_img_path, cfg.train_mask_path, transform=val_tf)
    val_ds.dataset = full_val_dataset # 替换底层 dataset 为无增强版本
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    
    # 2. 模型构建 - SOTA 核心
    # 使用 U-Net++ 架构，EfficientNet-B4 编码器
    print(f"Creating {cfg.arch} with {cfg.encoder} encoder...")
    model = smp.UnetPlusPlus(
        encoder_name=cfg.encoder,        
        encoder_weights=cfg.encoder_weights, 
        in_channels=cfg.in_channels,     
        classes=cfg.classes,             
        activation=None # 输出 logits
    )
    model.to(cfg.device)
    
    # 3. 损失函数和优化器
    # 使用 DiceLoss + BCELoss 的组合，这是分割任务的标准 SOTA Loss
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 4. 训练循环
    best_dice = 0.0
    
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}") as pbar:
            for imgs, masks in pbar:
                imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
                
                optimizer.zero_grad()
                logits = model(imgs)
                
                loss = loss_fn(logits, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
                logits = model(imgs)
                # 转为概率
                preds = torch.sigmoid(logits)
                # 二值化 [cite: 950]
                preds = (preds > 0.5).float()
                
                val_dice += dice_coef(preds, masks).item()
                
        avg_val_dice = val_dice / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Dice: {avg_val_dice:.4f}")
        
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_model.pth"))
            print(f"New Best Dice: {best_dice:.4f} (Saved)")
            
            # 可视化当前最好的结果
            # 取一个 batch 可视化
            imgs_vis = imgs.cpu().numpy()
            masks_vis = masks.cpu().numpy()
            preds_vis = preds.cpu().numpy()
            plot_results(imgs_vis, masks_vis, preds_vis, os.path.join(cfg.save_dir, f"vis_epoch_{epoch}.png"))

    print(f"Final Best Dice: {best_dice}")

if __name__ == "__main__":
    main()