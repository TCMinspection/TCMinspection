import os
from pathlib import Path

def sort_images_by_label(input_dir, output_root):
    """
    将形如 '001_yellow.jpg' 的文件根据颜色标签
    （red, black, white, yellow, cyan）
    分类移动至 output_root/标签名/ 目录下。

    Args:
        input_dir (str or Path): 包含原始图像的文件夹路径。
        output_root (str or Path): 分类后的根目录。
    """
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    allowed_labels = {"red", "black", "white", "yellow", "cyan"}

    for file in input_dir.glob("*.*"):
        if file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        parts = file.stem.split('_')
        if len(parts) < 2:
            continue
        label = parts[-1].lower()
        if label not in allowed_labels:
            continue  # 忽略未定义标签的文件
        dest_dir = output_root / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file.name
        file.rename(dest_path)

if __name__ == "__main__":
    # 示例用法：将 ./raw_faces 中的图像整理至 ./sorted_faces
    sort_images_by_label("C:/Users/陈祖彬/Desktop/计算机设计大赛项目/imagesource", "./sorted_faces")
