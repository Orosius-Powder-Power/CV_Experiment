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
    train_transform = A.Compose([
        # === SOTA 升级 2: 多尺度训练 ===
        # 在裁剪前，先随机缩放图片 (0.5 到 2.0 倍之间)
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=0),
        
        # 保证尺寸
        A.SmallestMaxSize(max_size=cfg.crop_size, always_apply=True),
        A.PadIfNeeded(min_height=cfg.crop_size, min_width=cfg.crop_size, border_mode=0, value=0, mask_value=255, always_apply=True),
        
        # 随机裁剪
        A.RandomCrop(width=cfg.crop_size, height=cfg.crop_size, always_apply=True),
        
        # 强力增强
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(p=0.3), # 增加颜色抖动
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3), # 模拟遮挡 (CutOut)
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=cfg.crop_size, always_apply=True),
        A.PadIfNeeded(min_height=cfg.crop_size, min_width=cfg.crop_size, border_mode=0, value=0, mask_value=255, always_apply=True),
        A.CenterCrop(width=cfg.crop_size, height=cfg.crop_size, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform