import torchvision
import os

# 确保目录存在
root_dir = "./data"
os.makedirs(root_dir, exist_ok=True)

print("正在下载 PASCAL VOC 2012 数据集 (约 2GB)...")
try:
    # download=True 会自动下载并解压
    dataset = torchvision.datasets.VOCSegmentation(
        root=root_dir, 
        year='2012', 
        image_set='train', 
        download=True
    )
    print("下载并解压完成！")
except Exception as e:
    print(f"下载失败: {e}")
    print("请尝试手动下载: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")
    print(f"解压后请确保路径结构为: {root_dir}/VOCdevkit/VOC2012/...")