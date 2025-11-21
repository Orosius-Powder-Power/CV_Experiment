import os
import urllib.request
import zipfile
import tifffile
import numpy as np
from PIL import Image

# 创建数据目录
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

print("正在下载 ISBI 2012 数据集...")
# 这里使用一个公开的 ISBI 2012 镜像源
url = "https://github.com/stardist/stardist/releases/download/0.1.0/isbi2012.zip"
zip_path = os.path.join(DATA_DIR, "isbi2012.zip")

if not os.path.exists(zip_path):
    urllib.request.urlretrieve(url, zip_path)
    print("下载完成，正在解压...")
else:
    print("文件已存在，跳过下载。")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)

# ISBI数据通常包含 train-volume.tif, train-labels.tif, test-volume.tif
# 确保文件名与文档一致 
files_check = ['train-volume.tif', 'train-labels.tif', 'test-volume.tif']
print(f"请检查 {DATA_DIR} 下是否包含以下文件: {files_check}")