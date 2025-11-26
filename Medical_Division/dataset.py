import torch
from torch.utils.data import Dataset
import tifffile
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISBIDataset(Dataset):
    def __init__(self, img_path, mask_path=None, transform=None):
        # 读取多页 TIFF [cite: 664]
        self.images = tifffile.imread(img_path)
        if mask_path:
            self.masks = tifffile.imread(mask_path)
        else:
            self.masks = None
            
        self.transform = transform
        
        print(f"Loaded data from {img_path}: {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        # 归一化到 [0, 1] 并在 transform 中标准化
        image = image.astype(np.float32) / 255.0
        
        if self.masks is not None:
            mask = self.masks[idx]
            # mask 是 0 或 255，转为 0 或 1 [cite: 950]
            mask = (mask / 255.0).astype(np.float32)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            # 增加 channel 维度 [1, H, W]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
                
            return image, mask
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image


#这里是提分的关键。Baseline只用了简单的翻转和裁剪 。我们要用 albumentations 加入弹性形变 (Elastic Transform)，这是医学图像分割的大杀器。
def get_transforms(cfg):
    train_transform = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # 弹性形变，模拟生物组织的形变
        # 新版 albumentations 将 alpha, sigma 整合进了参数列表，或者移除了 alpha_affine
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        
        # 增加 GridDistortion (网格畸变)，这也是医学图像常用的
        A.GridDistortion(p=0.5),
        
        # 增加 OpticalDistortion (光学畸变)
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        
        # 随机亮度对比度
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform