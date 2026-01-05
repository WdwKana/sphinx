import torch
import os

# ================= 配置部分 =================
# 原始数据文件路径 (请修改为你实际的 .pt 文件路径)
input_path = "/local/s4176650/sphinx/data/collect_RememberShapeAndColor3x2-v0.pt"

# 新的小数据集保存路径
output_path = "/local/s4176650/sphinx/data/collect_RememberShapeAndColor3x2-v0_500ep.pt"

# 目标 episode 数量
TARGET_EPISODES = 500
# ===========================================

print(f"Loading data from {input_path}...")
try:
    data = torch.load(input_path)
except FileNotFoundError:
    print(f"Error: File not found at {input_path}")
    exit(1)

# 定义需要处理的键 (基于 get_random_datasets_full_state.py 的输出)
keys_to_slice = ["obss", "states", "joints", "actions", "rewards", "masks"]

new_data = {}
original_count = 0

# 检查是否存在至少一个键来确定原始数据长度
if "obss" in data:
    original_count = data["obss"].shape[0]
else:
    # 尝试用其他键推断
    for key in keys_to_slice:
        if key in data:
            original_count = data[key].shape[0]
            break

print(f"Original dataset contains {original_count} episodes.")

if original_count < TARGET_EPISODES:
    print(f"Warning: Original dataset is smaller than target ({original_count} < {TARGET_EPISODES}). Copying full dataset.")
    TARGET_EPISODES = original_count

print(f"Slicing data to keep first {TARGET_EPISODES} episodes...")

for key, value in data.items():
    if key in keys_to_slice:
        if torch.is_tensor(value):
            # 假设第一维是 Episode (N, T, ...)
            new_data[key] = value[:TARGET_EPISODES]
            print(f"  Processed '{key}': {value.shape} -> {new_data[key].shape}")
        else:
            print(f"  Warning: Key '{key}' is in target list but is not a tensor. Copied as is.")
            new_data[key] = value
    else:
        # 对于不在列表中的其他元数据，直接复制
        print(f"  Copying metadata key '{key}' as is.")
        new_data[key] = value

print(f"\nSaving new dataset to {output_path}...")
torch.save(new_data, output_path)
print("Done!")