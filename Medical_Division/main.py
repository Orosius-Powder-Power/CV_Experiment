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

# 创新点4：预测时，把图分别水平翻转、垂直翻转、分别预测一次，然后把 3 次的结果取平均。这能极大提升稳定性。
def predict_with_tta(model, img):
    """
    手动实现 TTA: 原图 + 水平翻转 + 垂直翻转
    """
    # 1. 原图预测
    logits = model(img)
    preds = torch.sigmoid(logits)
    
    # 2. 水平翻转预测
    img_h = torch.flip(img, dims=[3])
    logits_h = model(img_h)
    preds_h = torch.sigmoid(logits_h)
    preds_h = torch.flip(preds_h, dims=[3]) # 翻转回来
    
    # 3. 垂直翻转预测
    img_v = torch.flip(img, dims=[2])
    logits_v = model(img_v)
    preds_v = torch.sigmoid(logits_v)
    preds_v = torch.flip(preds_v, dims=[2]) # 翻转回来
    
    # 平均
    return (preds + preds_h + preds_v) / 3.0

def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # 1. 数据准备
    # 数据预处理创新：Baseline只用了简单的翻转和裁剪 。我们要用 albumentations 加入弹性形变 (Elastic Transform)，这是医学图像分割的大杀器。
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
    # 创新点：使用 DiceLoss + BCELoss 的组合，这是分割任务的标准 SOTA Loss
    # BCELoss就是标准的交叉熵损失，DiceLoss使模型推断的主要部分与实际的交集更大
    # 作用: 解决正负样本不平衡。医学图像里，背景（黑色）通常很大，病灶/组织（白色）很小。如果只用 BCE，模型只要把所有像素都预测成黑色，准确率也能很高，但 DiceLoss 强迫模型必须让“交集”变大，必须预测准那个小白点。
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
                
                # 转为概率
                # logits = model(imgs)
                # preds = torch.sigmoid(logits)

                # 使用 TTA 替代直接预测
                preds = predict_with_tta(model, imgs)
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