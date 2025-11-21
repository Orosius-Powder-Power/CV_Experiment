import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def dice_coef(y_pred, y_true, smooth=1e-6):
    """
    计算 Dice 系数
    y_pred: 预测概率或二值化后的结果
    y_true: 真实标签
    """
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dice

def plot_results(images, masks, preds, save_path):
    """可视化:原图、GT、预测 [cite: 1225]"""
    count = len(images)
    plt.figure(figsize=(10, 4 * count))
    for i in range(count):
        plt.subplot(count, 3, i*3 + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title("Image")
        plt.axis('off')
        
        plt.subplot(count, 3, i*3 + 2)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        
        plt.subplot(count, 3, i*3 + 3)
        plt.imshow(preds[i].squeeze(), cmap='gray')
        plt.title("Prediction (U-Net++)")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()