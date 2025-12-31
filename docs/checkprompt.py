import json
import os
import ssl
import urllib.request

# 禁用SSL证书验证（解决macOS上的证书问题）
ssl._create_default_https_context = ssl._create_unverified_context

meta_url = "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/meta_data.json"
meta_path = "/tmp/mjhq_meta_data.json"

if not os.path.exists(meta_path):
    print("正在下载 meta_data.json ...")
    urllib.request.urlretrieve(meta_url, meta_path)
    print("下载完成！")

with open(meta_path, 'r') as f:
    meta = json.load(f)

print("=" * 60)
print("MJHQ-30K Prompt 查询工具")
print("输入文件名（不含扩展名）查询prompt，输入 'q' 退出")
print("=" * 60)

while True:
    filename = input("\n请输入文件名: ").strip()
    
    if filename.lower() == 'q':
        print("再见！")
        break
    
    # 去除可能的扩展名
    if filename.endswith('.png') or filename.endswith('.jpg'):
        filename = filename.rsplit('.', 1)[0]
    
    if filename in meta:
        print(f"\n文件名: {filename}")
        print(f"Prompt: {meta[filename]['prompt']}")
        print(f"类别: {meta[filename]['category']}")
    else:
        print(f"错误: 未找到文件名 '{filename}'，请检查输入是否正确")