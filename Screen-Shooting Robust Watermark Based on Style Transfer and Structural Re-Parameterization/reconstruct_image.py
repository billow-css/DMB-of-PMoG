import cv2
import numpy as np
import os


def reconstruct_image(blocks_folder, output_path, block_size=128):
    """
    将分割的块重新合成为完整图像，自动去除补零部分

    参数:
        blocks_folder: 包含分块的目录
        output_path: 输出图像路径
        block_size: 分块大小(默认128x128)
    """
    # 读取元数据
    meta_file = os.path.join(blocks_folder, "meta.txt")
    if not os.path.exists(meta_file):
        raise FileNotFoundError("找不到元数据文件meta.txt")

    with open(meta_file, "r") as f:
        h, w = map(int, f.read().split(","))

    print(f"将重建原始尺寸: {w}x{h}")

    # 获取所有块并按序号排序
    block_files = []
    for f in os.listdir(blocks_folder):
        if f.endswith(".png") and f[:-4].isdigit():
            block_files.append(f)

    block_files.sort(key=lambda x: int(x.split(".")[0]))

    if not block_files:
        raise ValueError("未找到有效的分块文件")

    # 计算补零后的尺寸
    padded_h = ((h + block_size - 1) // block_size) * block_size
    padded_w = ((w + block_size - 1) // block_size) * block_size

    # 创建空白画布
    reconstructed = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)

    # 重新组合图像
    blocks_per_row = padded_w // block_size
    for i, block_file in enumerate(block_files):
        row = i // blocks_per_row
        col = i % blocks_per_row

        y = row * block_size
        x = col * block_size

        block_path = os.path.join(blocks_folder, block_file)
        block = cv2.imread(block_path)

        if block is None:
            raise ValueError(f"无法读取块文件: {block_file}")

        reconstructed[y:y + block_size, x:x + block_size] = block

    # 裁剪补零部分
    final_image = reconstructed[:h, :w]

    # 保存结果
    cv2.imwrite(output_path, final_image)
    print(f"图像已重建并保存到 {output_path}")


# 使用示例
if __name__ == "__main__":
    reconstruct_image("C:/Users/admin/Desktop/qianru/", "reconstructed1.jpg")