import os
from pathlib import Path

def rename_yellow_to_cyan_with_prefix(image_dir, prefix="new"):
    image_dir = Path(image_dir)
    count = 0

    for file in image_dir.iterdir():
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            stem = file.stem.replace("yellow", "cyan")  # 替换 yellow 为 cyan
            new_name = prefix + stem + file.suffix      # 添加前缀 + 保留扩展名
            new_path = file.with_name(new_name)
            file.rename(new_path)
            count += 1

    print(f"✅ 已重命名 {count} 个文件（添加前缀 + 替换 yellow → cyan）")

if __name__ == "__main__":
    # 修改为你的实际输出目录
    target_folder = "D:/TCMinspection/Gan_output"
    rename_yellow_to_cyan_with_prefix(target_folder)
