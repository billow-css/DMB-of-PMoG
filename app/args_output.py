# -*- coding: utf-8 -*-
"""argparse：Lite / 输出 / 路径参数组。"""

from __future__ import annotations

import argparse

from app.constants import TOOL_MODE_CHOICES


def add_lite_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("训练规模（全量 / Lite）")
    scale = g.add_mutually_exclusive_group()
    scale.add_argument(
        "--lite", action="store_true", help="跳过菜单，直接启用 Lite（约 lite_ratio 比例样本）"
    )
    scale.add_argument(
        "--full", action="store_true", help="跳过菜单，直接使用全量 COCOMask"
    )
    g.add_argument(
        "--no_interactive",
        action="store_true",
        help="禁用启动菜单；未指定 --lite/--full 时默认全量",
    )
    g.add_argument(
        "--lite_ratio",
        type=float,
        default=0.15,
        help="Lite 抽样比例，建议 0.10~0.20 (默认: 0.15)",
    )
    g.add_argument("--lite_seed", type=int, default=42, help="Lite 抽样随机种子 (默认: 42)")
    g.add_argument(
        "--lite_auto_tune",
        action="store_true",
        default=True,
        help="Lite 时自动下调 batch_size / num_epoch（默认开启）",
    )
    g.add_argument(
        "--no_lite_auto_tune",
        action="store_false",
        dest="lite_auto_tune",
        help="禁用 Lite 自动下调超参",
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("输出与进度")
    g.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=[
            "train_mask",
            "eval_mask",
            "test_accuracy",
            "test_embedding",
            *TOOL_MODE_CHOICES,
            "log_query",
            "quit",
        ],
        help="运行模式；省略时交互面板（0–4 训练评估，5–9 工具，a–c 实验，l 日志）",
    )
    g.add_argument(
        "--log_run",
        type=str,
        default=None,
        help="配合 --mode log_query：直接调出该 run_id（或前缀）",
    )
    g.add_argument("--num_workers", type=int, default=1, help="DataLoader workers (默认: 1)")
    g.add_argument(
        "--log_step", type=int, default=40, help="文本日志间隔（无进度条时）(默认: 40)"
    )
    g.add_argument(
        "--model_save_step", type=int, default=1, help="每隔多少 epoch 存盘 (默认: 1)"
    )
    g.add_argument(
        "--quiet", action="store_true", help="减少控制台啰嗦输出，保留关键进度与结果"
    )
    g.add_argument("--verbose", action="store_true", help="打印更详细的中间信息")
    g.add_argument(
        "--no_progress",
        action="store_true",
        help="关闭 tqdm 进度条（适合 nohup / 重定向）",
    )
    g.add_argument(
        "--save_viz",
        action="store_true",
        default=True,
        help="训练结束每个 epoch 保存可视化拼图 (默认开启)",
    )
    g.add_argument(
        "--no_save_viz",
        action="store_false",
        dest="save_viz",
        help="关闭 epoch 可视化拼图导出",
    )
    g.add_argument(
        "--no_epoch_plot",
        action="store_true",
        help="关闭每 epoch 结束后的 msg/den/loss 曲线弹窗",
    )
    g.add_argument(
        "--no_train_editor",
        action="store_true",
        help="跳过训练前超参编辑窗（仍可用命令行参数）",
    )


def add_path_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("路径")
    g.add_argument(
        "--image_dir",
        type=str,
        default="Datasets/COCOMask/train/train_class/",
        help="训练图像目录；嵌入默认改为 Datasets/images/（可用本参数覆盖）",
    )
    g.add_argument(
        "--image_val_dir",
        type=str,
        default="Datasets/COCOMask/val/val_class/",
        help="验证目录；测准默认改为 Datasets/Recover/capture/",
    )
    g.add_argument(
        "--psnr_ref_dir",
        type=str,
        default="Datasets/images_watermarked/",
        help="拍屏测准 PSNR 参考目录（同索引含水印宿主，默认 images_watermarked/）",
    )
    g.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    g.add_argument("--model_save_dir", type=str, default="models", help="模型权重目录")
    g.add_argument(
        "--model_name", type=str, default="Encoder_Decoder_Model", help="权重文件名前缀"
    )
    g.add_argument("--result_dir", type=str, default="results", help="结果输出目录")
    g.add_argument(
        "--wmat_dir",
        type=str,
        default="results/WatermarkMatrix/w.mat",
        help="水印矩阵 .mat 路径",
    )
