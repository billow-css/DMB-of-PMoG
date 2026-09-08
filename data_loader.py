#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader.py
==============
PIMoG 数据加载模块。

职责
----
1. 训练数据：读取 COCOMask 拼接图（左：宿主图 / 右：mask），随机生成 30-bit 水印。
2. 测试数据：读取独立图像，并从 .mat 水印矩阵按文件名索引取消息比特。
3. Lite 模式：仅抽取 COCOMask 的 10%~20%（默认 15%）样本，用于小批量快速训练。

数据格式约定
------------
训练图：水平拼接，宽度 = 2 * image_size
  [ host (H×W×3) | mask (H×W×3) ]

测试图：普通 RGB 图像，文件名形如 ``{index}.png``（或 ``test{index}.png``），
index 对应 w.mat 行号。嵌入请用 ``Datasets/images/``，勿直接喂 COCOMask 拼接图。

公开接口
--------
- ImageLoader_for_train_mask  训练 Dataset
- ImageLoader_for_test        测试 / 嵌入 Dataset
- get_loader(...)             构建 DataLoader（支持 lite 子集）
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import scipy.io as scio
from torch.utils import data

from cli_utils import log_info, log_ok, log_warn


def _parse_wmat_index(filename: str) -> int:
    """
    从文件名解析 w.mat 行号。

    支持 ``0.png``、``12.jpg``；也兼容 ``test0.png``（取末尾连续数字）。
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem.isdigit() or (stem.startswith("-") and stem[1:].isdigit()):
        return int(stem)
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", stem)
    if m:
        return int(m.group(1))
    raise ValueError(
        f"无法从文件名解析水印索引（需形如 0.png 或 test12.png）: {filename}"
    )


# ---------------------------------------------------------------------------
# 工具：Lite 子集抽样
# ---------------------------------------------------------------------------
def _list_image_files(data_dir: str) -> List[str]:
    """列出目录下所有文件（保持 os.listdir 原始顺序，再排序以保证可复现）。"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    paths = sorted(os.listdir(data_dir))
    # 过滤隐藏文件 / 非图像
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    filtered = [
        p for p in paths
        if not p.startswith(".") and os.path.splitext(p)[1].lower() in exts
    ]
    if not filtered:
        # 回退：保留全部非隐藏文件（兼容原仓库行为）
        filtered = [p for p in paths if not p.startswith(".")]
    return filtered


def select_lite_subset(
    paths: Sequence[str],
    ratio: float = 0.15,
    seed: int = 42,
) -> List[str]:
    """
    从完整路径列表中随机抽取 Lite 子集。

    Parameters
    ----------
    paths : sequence of str
        完整样本文件名列表。
    ratio : float
        抽样比例，建议落在 [0.10, 0.20]；会自动夹紧到 (0, 1]。
    seed : int
        随机种子，保证同一数据集可复现。

    Returns
    -------
    list of str
        抽样后的文件名列表（已排序）。
    """
    ratio = float(ratio)
    if ratio <= 0 or ratio > 1:
        log_warn(f"lite_ratio={ratio} 非法，已夹紧到 (0, 1]")
        ratio = min(max(ratio, 1e-6), 1.0)

    n_total = len(paths)
    n_keep = max(1, int(round(n_total * ratio)))
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, size=n_keep, replace=False)
    subset = sorted(paths[i] for i in indices)
    log_ok(
        f"Lite 子集: {n_keep}/{n_total} "
        f"({100.0 * n_keep / max(n_total, 1):.1f}%, seed={seed})"
    )
    return subset


# ---------------------------------------------------------------------------
# Dataset：训练（mask）
# ---------------------------------------------------------------------------
class ImageLoader_for_train_mask(data.Dataset):
    """
    COCOMask 训练数据集。

    每张图水平拼接 [host | mask]；消息比特默认每次随机采样。
    评估（``message_seed`` 非空）时按 ``seed + index`` 可复现采样，
    保证同模型换噪声层时 PSNR（Encoded vs Host）完全一致。

    Parameters
    ----------
    data_dir : str
        图像目录（末尾建议带 /）。
    image_size : int
        宿主图边长（mask 同尺寸）。
    img_paths : list of str, optional
        若给定则直接使用该文件列表（Lite 模式传入子集）。
    transform : callable, optional
        预留，当前未使用。
    message_seed : int, optional
        若给定，则消息由 ``RandomState(seed + index)`` 生成（可复现）；
        ``None`` 则每次 ``__getitem__`` 全局随机（训练用）。
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int,
        img_paths: Optional[List[str]] = None,
        transform=None,
        message_seed: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir if data_dir.endswith(("/", "\\")) else data_dir + os.sep
        self.transform = transform
        self.image_size = image_size
        self.img_paths = img_paths if img_paths is not None else _list_image_files(self.data_dir)
        self.message_seed = message_seed

    def __len__(self) -> int:
        return len(self.img_paths)

    def _sample_message(self, index: int) -> np.ndarray:
        if self.message_seed is None:
            m = np.random.rand(30)
        else:
            rng = np.random.RandomState(int(self.message_seed) + int(index) * 10007)
            m = rng.rand(30)
        m = (m >= 0.5).astype(np.float64)
        return m

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        Data_img : float32 [3, H, W]  归一化到 [-1, 1]
        m        : float64 [30]       0/1 水印比特
        mask     : float32 [3, H, W]  视觉感知权重 mask
        """
        imagesize = self.image_size
        curr_img_path = self.img_paths[index]
        img = cv2.imread(self.data_dir + curr_img_path, 1)
        if img is None:
            raise RuntimeError(f"无法读取图像: {self.data_dir + curr_img_path}")

        data_img = img[:, :imagesize, :]
        mask = img[:, imagesize:, :]
        mask = np.float32(mask) / 255.0
        mask = (mask.transpose((2, 0, 1)) + 1) * 3

        m = self._sample_message(index)

        Data_img = data_img.transpose((2, 0, 1))
        Data_img = np.float32(Data_img / 255 * 2 - 1)
        return Data_img, m, mask


# ---------------------------------------------------------------------------
# Dataset：测试 / 嵌入
# ---------------------------------------------------------------------------
class ImageLoader_for_test(data.Dataset):
    """
    测试 / 水印嵌入数据集。

    从 ``w.mat`` 读取固定水印矩阵，按文件名前缀数字索引消息行。

    Parameters
    ----------
    data_dir : str
        图像目录。
    image_size : int
        缩放到的边长。
    w_path : str
        水印矩阵 .mat 路径（键名 ``w``）。
    img_paths : list of str, optional
        Lite / 自定义子集。
    transform : callable, optional
        预留。
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int,
        w_path: str,
        img_paths: Optional[List[str]] = None,
        transform=None,
    ):
        super().__init__()
        self.data_dir = data_dir if data_dir.endswith(("/", "\\")) else data_dir + os.sep
        self.transform = transform
        self.image_size = image_size
        self.img_paths = img_paths if img_paths is not None else _list_image_files(self.data_dir)
        W = scio.loadmat(w_path)
        self.w = W["w"]

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Returns
        -------
        Data_img : float32 [3, H, W]
        m        : 水印比特向量（来自 w.mat）
        num      : 文件名前缀整数索引
        """
        imagesize = self.image_size
        curr_img_path = self.img_paths[index]
        num = _parse_wmat_index(curr_img_path)
        if num < 0 or num >= self.w.shape[0]:
            raise IndexError(
                f"文件 {curr_img_path} 索引 {num} 超出 w.mat 行数 "
                f"{self.w.shape[0]}（合法 0..{self.w.shape[0] - 1}）"
            )
        img = cv2.imread(self.data_dir + curr_img_path, 1)
        if img is None:
            raise RuntimeError(f"无法读取图像: {self.data_dir + curr_img_path}")

        data_img = cv2.resize(img, (imagesize, imagesize))
        m = self.w[num, :]
        Data_img = data_img.transpose((2, 0, 1))
        Data_img = np.float32(Data_img / 255 * 2 - 1)
        return Data_img, m, num


# ---------------------------------------------------------------------------
# DataLoader 工厂
# ---------------------------------------------------------------------------
def get_loader(
    image_dir: str,
    image_size: int = 128,
    batch_size: int = 32,
    dataset: str = "train_mask",
    mode: str = "train_mask",
    num_workers: int = 1,
    w_path: str = "results/WatermarkMatrix/w.mat",
    lite: bool = False,
    lite_ratio: float = 0.15,
    lite_seed: int = 42,
):
    """
    构建并返回 ``torch.utils.data.DataLoader``。

    Parameters
    ----------
    image_dir : str
        图像根目录。
    image_size : int
        宿主图边长，默认 128。
    batch_size : int
        mini-batch 大小。
    dataset : str
        ``train_mask`` | ``eval_mask`` | ``test_accuracy`` | ``test_embedding``。
    mode : str
        运行模式；训练相关模式会开启 shuffle。
    num_workers : int
        DataLoader 工作进程数。
    w_path : str
        测试用水印矩阵路径。
    lite : bool
        True 时仅使用 ``lite_ratio`` 比例的数据子集。
    lite_ratio : float
        Lite 抽样比例，推荐 0.10–0.20。
    lite_seed : int
        Lite 抽样随机种子。

    Returns
    -------
    DataLoader
    """
    # 规范化目录分隔符
    if image_dir and not image_dir.endswith(("/", "\\")):
        image_dir = image_dir + os.sep

    img_paths = None
    if lite:
        all_paths = _list_image_files(image_dir)
        img_paths = select_lite_subset(all_paths, ratio=lite_ratio, seed=lite_seed)
        log_info(f"[{dataset}] Lite 加载: {image_dir} → {len(img_paths)} 张")
    else:
        n = len(_list_image_files(image_dir)) if os.path.isdir(image_dir) else 0
        log_info(f"[{dataset}] 全量加载: {image_dir} → {n} 张")

    if dataset in ("train_mask", "eval_mask"):
        # eval_mask：消息按 index 固定，避免「换噪声层 PSNR 差 0.01」的假象
        # （PSNR 只看 Encoded vs Host，本就与 Noiser 无关；随机消息会让两次评估略有差）
        msg_seed = 0 if (dataset == "eval_mask" or mode == "eval_mask") else None
        ds = ImageLoader_for_train_mask(
            image_dir,
            image_size,
            img_paths=img_paths,
            message_seed=msg_seed,
        )
        if msg_seed is not None:
            log_info(f"[{dataset}] 评估消息可复现（message_seed={msg_seed}）")
    elif dataset in ("test_accuracy", "test_embedding"):
        ds = ImageLoader_for_test(image_dir, image_size, w_path, img_paths=img_paths)
    else:
        raise ValueError(f"未知 dataset: {dataset}")

    # 原代码 shuffle=(mode=='train') 恒为 False；此处对训练模式正确开启
    do_shuffle = mode in ("train_mask", "train")

    loader = data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    return loader
