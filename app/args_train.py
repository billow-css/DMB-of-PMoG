# -*- coding: utf-8 -*-
"""argparse：模型 / 训练 / 噪声相关参数组。"""

from __future__ import annotations

import argparse


def add_model_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("模型")
    g.add_argument("--image_size", type=int, default=128, help="宿主图边长 (默认: 128)")
    g.add_argument("--num_channels", type=int, default=64, help="判别器通道数 (默认: 64)")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("训练")
    g.add_argument(
        "--dataset",
        type=str,
        default="train_mask",
        choices=["train_mask", "eval_mask", "test_accuracy", "test_embedding"],
        help="数据集 / 任务类型",
    )
    g.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="mini-batch 大小 (默认: 16；Lite 可自动下调)",
    )
    g.add_argument("--lambda1", type=float, default=3, help="消息损失权重 λ1（官方默认 3）")
    g.add_argument("--lambda2", type=float, default=1, help="图像/感知损失权重 λ2（官方默认 1）")
    g.add_argument(
        "--lambda3", type=float, default=0.001, help="GAN 损失权重 λ3（官方默认 0.001）"
    )
    g.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam 学习率（默认: 1e-3，与官方 Adam 默认一致）",
    )
    g.add_argument(
        "--num_epoch", type=int, default=100, help="总 epoch 数 (默认: 100；Lite 可自动下调)"
    )
    g.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
        help="Identity 预热 epoch 数（默认：全量=1，Lite=5；显式传 0 可关闭）",
    )
    g.add_argument(
        "--embed_strength",
        type=float,
        default=0.0,
        help="嵌入方式：0=论文整图输出（Encoder 直接出含水印图）；"
        ">0 时为残差嵌入 x+strength*tanh(r)（默认: 0）",
    )
    g.add_argument(
        "--embedding_epoch",
        type=int,
        default=0,
        help="续训起始 epoch / 评估加载的权重编号 (默认: 0=从头)",
    )
    g.add_argument(
        "--eval_ckpt",
        type=str,
        default=None,
        choices=["99", "ss_best", "best", "mb_best", "custom"],
        help="命名权重标签（兼容）；交互评估改为弹窗选文件。"
        "99 / ss_best / best；custom=配合 --weight_path",
    )
    g.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="直接指定评估/嵌入/测准权重 .pth（优先于 --eval_ckpt）",
    )
    g.add_argument(
        "--init_from_epoch",
        type=int,
        default=None,
        help="训练时拷贝并加载该 epoch 预训练权重作初始化（如 99）；与从头训练互斥",
    )
    g.add_argument(
        "--init_from_ckpt",
        type=str,
        default=None,
        choices=["99", "ss_best", "best", "mb_best"],
        help="训练初始化命名权重：99 / ss_best / best（优先于仅写 epoch）",
    )
    g.add_argument(
        "--distortion",
        type=str,
        default=None,
        choices=["Identity", "ScreenShooting", "ScreenShootingMB"],
        help="噪声层：ScreenShooting=基线；ScreenShootingMB=DMB 动态运动模糊；省略则交互选择",
    )
