import logging
import os
import matplotlib.pyplot as plt
import numpy as np

def setup_logger(log_dir, log_file):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger("CIFAR10_Exp")
    logger.setLevel(logging.INFO)
    
    # 防止重复打印
    if not logger.handlers:
        # 文件处理器
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file), mode='w')
        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger

def plot_training_results(train_losses, train_accs, test_accs, save_path):
    """
    可视化Loss和Accuracy,对应文档要求的上下子图
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 8))
    
    # 上图：Loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'r-', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 下图：Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, test_accs, 'g-', label='Test Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")