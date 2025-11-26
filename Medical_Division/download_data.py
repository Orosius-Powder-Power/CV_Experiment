import os
import urllib.request
import sys

# 确保保存目录与 config.py 中的路径一致
DATA_DIR = "./data/isbi2012"
os.makedirs(DATA_DIR, exist_ok=True)

# 使用 zhixuhao/unet 仓库的源，这是最常用的备份源
# 注意：GitHub Raw 链接在国内服务器可能会偶尔连接失败
BASE_URL = "https://github.com/zhixuhao/unet/raw/master/data/"
FILES = ["train-volume.tif", "train-labels.tif", "test-volume.tif"]

print(f"正在准备下载数据到: {DATA_DIR} ...")

def download_file(filename):
    url = BASE_URL + filename
    save_path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(save_path):
        print(f"[跳过] {filename} 已存在。")
        return True
        
    print(f"[下载中] {filename} 从 {url} ...")
    try:
        # 添加 User-Agent 模拟浏览器，防止被简单的反爬拦截
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, save_path)
        print(f"[成功] {filename} 下载完成。")
        return True
    except Exception as e:
        print(f"[失败] 下载 {filename} 出错: {e}")
        return False

success_count = 0
for f in FILES:
    if download_file(f):
        success_count += 1

if success_count == 3:
    print("\n所有数据下载成功！可以直接运行 main.py 了。")
else:
    print("\n部分文件下载失败。")
    print("========================================")
    print("【备用方案】如果服务器网络无法访问 GitHub，请尝试手动下载：")
    print("1. 在你自己的电脑上访问: https://github.com/zhixuhao/unet/tree/master/data")
    print("2. 下载 train-volume.tif, train-labels.tif, test-volume.tif 三个文件")
    print(f"3. 将它们上传到服务器的目录: {os.path.abspath(DATA_DIR)}")
    print("========================================")