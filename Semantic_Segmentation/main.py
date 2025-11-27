import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

from config import Config
from dataset import VOCDataset, get_transforms

# 设置 matplotlib 后端，防止在无显示器的服务器上报错
plt.switch_backend('Agg')

# ================= 工具类 =================

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

class ConfusionMatrix(object):
    """SOTA 标准：构建全局混淆矩阵来计算 mIoU"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (b >= 0) & (b < n)
            inds = n * b[k].to(torch.int64) + a[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
    def compute(self):
        if self.mat is None: return 0.0
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return iu.mean().item() * 100

def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power=0.9):
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# ================= 新增核心函数 =================

def predict_with_tta(model, imgs):
    """
    SOTA 级 TTA: 多尺度 + 翻转 (Multi-Scale Flip Inference)
    Scales: [0.75, 1.0, 1.25]
    """
    scales = [0.75, 1.0, 1.25] # 经典的 3 尺度
    b, c, h, w = imgs.shape
    final_probs = torch.zeros((b, 21, h, w), device=imgs.device)
    
    for scale in scales:
        # 1. 缩放输入
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            # 确保尺寸可以被 16 整除 (DeepLab 要求)
            new_h = ((new_h - 1) // 16 + 1) * 16
            new_w = ((new_w - 1) // 16 + 1) * 16
            input_tensor = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear', align_corners=True)
        else:
            input_tensor = imgs
            
        # 2. 预测 (原图)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # 3. 预测 (水平翻转)
        input_flip = torch.flip(input_tensor, dims=[3])
        logits_flip = model(input_flip)
        probs_flip = torch.softmax(logits_flip, dim=1)
        probs_flip = torch.flip(probs_flip, dims=[3])
        
        # 融合当前尺度的结果
        current_scale_probs = (probs + probs_flip) / 2.0
        
        # 4. 还原尺寸 (如果是缩放过的)
        if scale != 1.0:
            current_scale_probs = F.interpolate(current_scale_probs, size=(h, w), mode='bilinear', align_corners=True)
            
        final_probs += current_scale_probs
        
    # 取平均
    final_probs /= len(scales)
    return final_probs


def get_voc_palette(num_classes=21):
    """生成 PASCAL VOC 的标准颜色表，用于可视化"""
    n = num_classes
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def denormalize(img_tensor):
    """反归一化，将 tensor 转回可视化的 numpy 图片"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def visualize_results(model, loader, device, save_dir, epoch, num_samples=3):
    """保存 原图 | GT | 预测图 的对比效果"""
    model.eval()
    palette = get_voc_palette(21)
    
    # 取一个 batch
    imgs, masks = next(iter(loader))
    imgs, masks = imgs.to(device), masks.to(device)
    
    with torch.no_grad():
        probs = predict_with_tta(model, imgs)
        preds = torch.argmax(probs, dim=1)
    
    # 绘图
    plt.figure(figsize=(12, 4 * num_samples))
    for i in range(min(num_samples, len(imgs))):
        # 1. 原图
        img_np = denormalize(imgs[i])
        
        # 2. GT Mask
        mask_np = masks[i].cpu().numpy().astype(np.uint8)
        mask_pil = Image.fromarray(mask_np)
        mask_pil.putpalette(palette)
        mask_rgb = np.array(mask_pil.convert('RGB'))
        
        # 3. Pred Mask
        pred_np = preds[i].cpu().numpy().astype(np.uint8)
        pred_pil = Image.fromarray(pred_np)
        pred_pil.putpalette(palette)
        pred_rgb = np.array(pred_pil.convert('RGB'))
        
        # 组合显示
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img_np); plt.title("Image"); plt.axis('off')
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(mask_rgb); plt.title("Ground Truth"); plt.axis('off')
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(pred_rgb); plt.title(f"Prediction (Epoch {epoch})"); plt.axis('off')
        
    save_path = os.path.join(save_dir, f"vis_result_epoch_{epoch}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

def plot_history(train_losses, val_mious, save_dir):
    """绘制训练曲线：Loss 和 mIoU"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 6))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    # mIoU 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_mious, 'b-', label='Val mIoU')
    plt.title('Validation mIoU')
    plt.xlabel('Epochs'); plt.ylabel('mIoU (%)'); plt.legend(); plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()

# ================= 主函数 =================

def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # 1. 数据
    train_tf, val_tf = get_transforms(cfg)
    train_ds = VOCDataset(root=cfg.data_root, year='2012', image_set='train', download=False, transform=train_tf)
    val_ds = VOCDataset(root=cfg.data_root, year='2012', image_set='val', download=False, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    # 2. 模型
    print(f"Building SOTA Model: {cfg.arch} + {cfg.encoder}...")
    model = smp.DeepLabV3Plus(
        encoder_name=cfg.encoder, 
        encoder_weights=cfg.encoder_weights, 
        in_channels=3, 
        classes=cfg.num_classes
    )
    model.to(cfg.device)
    
    # 3. 优化器 & 学习率
    init_lr = cfg.learning_rate 
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255) 
    
    best_miou = 0.0
    total_iters = cfg.epochs * len(train_loader)
    curr_iter = 0
    
    # 记录历史数据
    history_loss = []
    history_miou = []

    # 4. 训练循环
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for imgs, masks in pbar:
            poly_lr_scheduler(optimizer, init_lr, curr_iter, total_iters)
            curr_iter += 1
            
            imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), imgs.size(0))
            pbar.set_postfix(loss=train_loss.avg, lr=optimizer.param_groups[0]['lr'])
        
        # 记录 Loss
        history_loss.append(train_loss.avg)
            
        # 验证阶段 (使用 TTA + 全局混淆矩阵)
        model.eval()
        conf_mat = ConfusionMatrix(cfg.num_classes)
        
        print("Evaluating with TTA...")
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="Val"):
                imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
                
                # 调用独立的 TTA 函数
                probs = predict_with_tta(model, imgs)
                preds = torch.argmax(probs, dim=1)
                
                conf_mat.update(preds, masks)
        
        val_miou = conf_mat.compute()
        history_miou.append(val_miou)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss.avg:.4f} | Val mIoU: {val_miou:.2f}%")
        
        # 保存最佳模型并可视化
        if val_miou > best_miou:
            best_miou = val_miou

            print(f"炼丹ing~ New Best mIoU: {best_miou:.2f}% (Saved)")
            
            if epoch > 100 or val_miou > 75.0:
                # 生成效果图
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_model.pth"))
                visualize_results(model, val_loader, cfg.device, cfg.save_dir, epoch)
            
        # 每个 epoch 结束都更新曲线图
        plot_history(history_loss, history_miou, cfg.save_dir)

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "last_model.pth"))
    print(f"Final Best mIoU: {best_miou:.2f}%")

if __name__ == "__main__":
    main()