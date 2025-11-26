import os
import torch
import numpy as np
from torchvision.datasets import VOCSegmentation
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VOCDataset(VOCSegmentation):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None):
        super().__init__(root, year=year, image_set=image_set, download=download)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        img_np = np.array(img)
        target_np = np.array(target)

        if self.transform:
            augmented = self.transform(image=img_np, mask=target_np)
            img_tensor = augmented['image']
            target_tensor = augmented['mask']
        else:
            # Fallback
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            target_tensor = torch.from_numpy(target_np).long()

        # target 需要是 long 类型 (int64) 用于 CrossEntropy
        return img_tensor, target_tensor.long()

def get_transforms(cfg):
    # 强力数据增强
    train_transform = A.Compose([
        # 1. 先把短边缩放到至少 512，确保图片够大
        A.SmallestMaxSize(max_size=cfg.crop_size, always_apply=True),
        
        # 2. 如果长宽还有一边小于 512 (极少数情况)，用 Pad 补齐
        # value=0 (黑色填充), mask_value=255 (忽略标签填充)
        A.PadIfNeeded(min_height=cfg.crop_size, min_width=cfg.crop_size, 
                      border_mode=0, value=0, mask_value=255, always_apply=True),
        
        # 3. 现在图片肯定 >= 512x512 了，可以放心裁切（随机裁剪）
        A.RandomCrop(width=cfg.crop_size, height=cfg.crop_size, always_apply=True),
        
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # 验证集只做 Resize/CenterCrop 和 归一化
    val_transform = A.Compose([
        # 验证集也做同样的尺寸保证
        A.SmallestMaxSize(max_size=cfg.crop_size, always_apply=True),
        A.PadIfNeeded(min_height=cfg.crop_size, min_width=cfg.crop_size, 
                      border_mode=0, value=0, mask_value=255, always_apply=True),
        A.CenterCrop(width=cfg.crop_size, height=cfg.crop_size, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform