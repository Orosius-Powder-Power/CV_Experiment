import os
import urllib.request
import ssl
import sys
import time

# 1. 配置下载目录
DATA_DIR = "./data/isbi2012"
os.makedirs(DATA_DIR, exist_ok=True)

# 2. 配置加速镜像 (这是关键！能解决服务器连不上 GitHub 的问题)
# 使用 mirror.ghproxy.com 代理下载
PROXY_PREFIX = "https://mirror.ghproxy.com/"

# 3. 正确的文件源 (ShawnBIT 仓库确认包含 tif 文件)
BASE_REPO = "https://github.com/ShawnBIT/UNet-family/raw/master/data/ISBI/"
FILES = [
    "train-volume.tif",
    "train-labels.tif",
    "test-volume.tif"
]

# 忽略 SSL 证书验证 (防止服务器证书报错)
ssl._create_default_https_context = ssl._create_unverified_context

def report_progress(block_num, block_size, total_size):
    """显示下载进度条"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        bar = '#' * int(percent / 2)
        print(f"\r下载进度: [{bar:<50}] {percent:.2f}%", end='')
    else:
        print(f"\r已下载: {downloaded / 1024 / 1024:.2f} MB", end='')

def download_with_retry(filename):
    # 拼接加速链接
    # 最终长这样: https://mirror.ghproxy.com/https://github.com/...
    original_url = BASE_REPO + filename
    accelerated_url = PROXY_PREFIX + original_url
    save_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(save_path):
        # 简单检查一下大小，防止断点续传导致的损坏文件 (ISBI tif 大约是 1-2MB 左右，甚至更小)
        if os.path.getsize(save_path) > 1000: 
            print(f"[跳过] {filename} 已存在且大小正常。")
            return True
        else:
            print(f"[警告] {filename} 存在但文件过小，重新下载。")
            os.remove(save_path)

    print(f"\n正在下载: {filename}")
    print(f"源地址: {accelerated_url}")
    
    try:
        # 添加 User-Agent 伪装成浏览器
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)
        
        start_time = time.time()
        urllib.request.urlretrieve(accelerated_url, save_path, report_progress)
        end_time = time.time()
        
        print(f"\n[成功] {filename} 下载完成 (耗时 {end_time - start_time:.2f}s)")
        return True
    except Exception as e:
        print(f"\n[失败] 无法下载 {filename}。错误信息: {e}")
        # 如果加速链接失效，尝试一次直连
        print("尝试不使用加速代理直连...")
        try:
            urllib.request.urlretrieve(original_url, save_path, report_progress)
            print(f"\n[成功] {filename} (直连) 下载完成")
            return True
        except Exception as e2:
            print(f"\n[彻底失败] 直连也失败了: {e2}")
            return False

print(">>> 开始高速下载 ISBI 2012 数据集 <<<")
print(f"保存路径: {os.path.abspath(DATA_DIR)}")

success_count = 0
for f in FILES:
    if download_with_retry(f):
        success_count += 1

print("\n" + "="*30)
if success_count == len(FILES):
    print("恭喜！所有数据下载完成！请直接运行 main.py")
else:
    print(f"下载完成了 {success_count}/{len(FILES)} 个文件。如有失败，请检查网络或重试。")