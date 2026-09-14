import cv2
import numpy as np
import os


def split_image(input_path, output_folder, block_size=128):
    """
    将输入图像分割为128x128的小块，不足部分补零
    并按顺序命名为1.png, 2.png, 3.png...

    参数:
        input_path: 输入图像路径
        output_folder: 分块保存目录
        block_size: 分块大小(默认128x128)
    """
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径")

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 获取原始尺寸
    h, w = img.shape[:2]
    print(f"原始图像尺寸: {w}x{h}")

    # 计算需要补零的像素数
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    # 执行补零 (底部和右侧补零)
    padded_img = cv2.copyMakeBorder(img,
                                    0, pad_h,
                                    0, pad_w,
                                    cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])

    # 计算分块数量
    num_blocks_vert = padded_img.shape[0] // block_size
    num_blocks_horz = padded_img.shape[1] // block_size
    total_blocks = num_blocks_vert * num_blocks_horz
    print(f"将分割为 {total_blocks} 个 {block_size}x{block_size} 的块")

    # 分割并保存
    block_count = 1
    for y in range(0, padded_img.shape[0], block_size):
        for x in range(0, padded_img.shape[1], block_size):
            block = padded_img[y:y + block_size, x:x + block_size]
            output_path = os.path.join(output_folder, f"{block_count}.png")
            cv2.imwrite(output_path, block)
            block_count += 1

    # 保存元数据(用于后续重建)
    with open(os.path.join(output_folder, "meta.txt"), "w") as f:
        f.write(f"{h},{w}")  # 原始高度,原始宽度

    print(f"分割完成，块已保存到 {output_folder}")


# 使用示例
if __name__ == "__main__":
    split_image("C:/Users/admin/Desktop/ppt1.jpg", "C:/Users/admin/Desktop/cut1/")