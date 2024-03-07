import os
import sys
import requests
from tqdm import tqdm

# 检查参数是否正确
if len(sys.argv) != 2:
    print('请以模型名称作为参数，例如：download_model.py 124M')
    sys.exit(1)

# 获取模型名称
model = sys.argv[1]

# 创建模型存储目录
subdir = os.path.join('models', model)
if not os.path.exists(subdir):
    os.makedirs(subdir)
subdir = subdir.replace('\\', '/')  # 适用于 Windows

# 需要下载的文件列表
files = ['checkpoint', 'encoder.json', 'hparams.json', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']

# 下载文件
for filename in files:
    # 发起请求下载文件
    r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/" + subdir + "/" + filename, stream=True)

    # 写入文件
    with open(os.path.join(subdir, filename), 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = 1000
        with tqdm(ncols=100, desc="正在下载 " + filename, total=file_size, unit_scale=True) as pbar:
            # 以 1k 为单位进行写入，因为以太网数据包大小约为 1500 字节
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)
