#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======
DMB of PMoG 命令行入口（Dynamic Motion Blur of PMoG）。

命名说明
--------
相对原 PIMoG（Perspective / Illumination / Moire / …）少了光照验证的 ``I``，
写作 **PMoG**；本仓库主推噪声层为 **DMB**（Dynamic Motion Blur）。

谱系论文
--------
Fang, Han, et al. "PIMoG: An Effective Screen-shooting Noise-Layer Simulation
for Deep-Learning-Based Watermarking Network." ACM MM 2022.

四种核心运行模式
----------------
0. eval_mask       跳过训练，加载已有权重评估 Acc/BER/PSNR + 可视化
1. train_mask      端到端训练 Encoder–Noise–Decoder（含 GAN）
2. test_embedding  使用预训练模型嵌入水印并导出可视化拼图
3. test_accuracy   对矫正后的拍屏图评估 Acc/BER，并相对含水印宿主算 PSNR

交互功能面板另含工具（导出宿主 / PSNR / SSIM / 裁切）与拍屏实验入口。

常用示例
--------
python main.py
python main.py --mode eval_mask --no_interactive --eval_ckpt best --distortion ScreenShootingMB
python main.py --mode tool_verify_psnr --no_interactive
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from ckpt_utils import (
    ckpt_default_distortion,
    ckpt_label,
    ckpt_run_tag,
    ckpt_short,
    default_noise_menu_choice,
    named_best_filename,
    normalize_ckpt_tag,
    read_embed_strength_meta,
    resolve_ckpt_path,
)
from cli_utils import (
    PROJECT_FULL,
    PROJECT_SHORT,
    log_err,
    log_info,
    log_ok,
    log_warn,
    print_banner,
    print_config_summary,
    print_lite_badge,
    print_mode_badge,
    prompt_eval_ckpt,
    prompt_log_browser,
    prompt_noise_layer,
    prompt_run_mode,
    prompt_train_init,
    prompt_train_scale,
)

# 重依赖（torch / solver）延迟到 main()，保证 `python main.py -h` 无需 GPU 环境

REPO_ROOT = Path(__file__).resolve().parent

TOOL_MODE_CHOICES = (
    "tool_export_hosts",
    "tool_verify_psnr",
    "tool_verify_ssim",
    "tool_verify_both",
    "tool_crop_panels",
    "tool_batch_rectify",
    "tool_rectify_gui",
    "tool_server_gui",
)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """构建分组清晰、带 epilog 示例的命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            f"{PROJECT_SHORT} — {PROJECT_FULL} · "
            "screen-shooting robust watermark training & evaluation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互功能面板（推荐）
  python main.py

  # 跳过训练直接评估 mask_99
  python main.py --mode eval_mask --no_interactive --eval_ckpt 99 --embed_strength 0

  # 评估 ScreenShooting_best
  python main.py --mode eval_mask --no_interactive --eval_ckpt ss_best --distortion ScreenShooting

  # 评估 DMB_best (ScreenShootingMB)
  python main.py --mode eval_mask --no_interactive --eval_ckpt best --distortion ScreenShootingMB

  # Lite / 全量训练
  python main.py --mode train_mask --lite --distortion ScreenShootingMB
  python main.py --mode train_mask --full

  # 工具：嵌入 PSNR / SSIM
  python main.py --mode tool_verify_psnr --no_interactive
  python main.py --mode tool_verify_both --no_interactive
""".strip(),
    )

    # ---- 模型 ----
    g_model = parser.add_argument_group("模型")
    g_model.add_argument(
        "--image_size", type=int, default=128, help="宿主图边长 (默认: 128)"
    )
    g_model.add_argument(
        "--num_channels", type=int, default=64, help="判别器通道数 (默认: 64)"
    )

    # ---- 训练 ----
    g_train = parser.add_argument_group("训练")
    g_train.add_argument(
        "--dataset",
        type=str,
        default="train_mask",
        choices=["train_mask", "eval_mask", "test_accuracy", "test_embedding"],
        help="数据集 / 任务类型",
    )
    g_train.add_argument(
        "--batch_size", type=int, default=2, help="mini-batch 大小 (默认: 16；Lite 可自动下调)"
    )
    g_train.add_argument(
        "--lambda1", type=float, default=3,
        help="消息损失权重 λ1（官方默认 3）",
    )
    g_train.add_argument(
        "--lambda2", type=float, default=1,
        help="图像/感知损失权重 λ2（官方默认 1）",
    )
    g_train.add_argument(
        "--lambda3", type=float, default=0.001, help="GAN 损失权重 λ3（官方默认 0.001）"
    )
    g_train.add_argument(
        "--lr", type=float, default=1e-3, help="Adam 学习率（默认: 1e-3，与官方 Adam 默认一致）"
    )
    g_train.add_argument(
        "--num_epoch", type=int, default=100, help="总 epoch 数 (默认: 100；Lite 可自动下调)"
    )
    g_train.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
        help="Identity 预热 epoch 数（默认：全量=1，Lite=5；显式传 0 可关闭）",
    )
    g_train.add_argument(
        "--embed_strength",
        type=float,
        default=0.0,
        help="嵌入方式：0=论文整图输出（Encoder 直接出含水印图）；"
        ">0 时为残差嵌入 x+strength*tanh(r)（默认: 0）",
    )
    g_train.add_argument(
        "--embedding_epoch",
        type=int,
        default=0,
        help="续训起始 epoch / 评估加载的权重编号 (默认: 0=从头)",
    )
    g_train.add_argument(
        "--eval_ckpt",
        type=str,
        default=None,
        choices=["99", "ss_best", "best", "mb_best", "custom"],
        help="命名权重标签（兼容）；交互评估改为弹窗选文件。"
        "99 / ss_best / best；custom=配合 --weight_path",
    )
    g_train.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="直接指定评估/嵌入/测准权重 .pth（优先于 --eval_ckpt）",
    )
    g_train.add_argument(
        "--init_from_epoch",
        type=int,
        default=None,
        help="训练时拷贝并加载该 epoch 预训练权重作初始化（如 99）；与从头训练互斥",
    )
    g_train.add_argument(
        "--init_from_ckpt",
        type=str,
        default=None,
        choices=["99", "ss_best", "best", "mb_best"],
        help="训练初始化命名权重：99 / ss_best / best（优先于仅写 epoch）",
    )
    g_train.add_argument(
        "--distortion",
        type=str,
        default=None,
        choices=["Identity", "ScreenShooting", "ScreenShootingMB"],
        help="噪声层：ScreenShooting=基线；ScreenShootingMB=DMB 动态运动模糊；省略则交互选择",
    )

    # ---- Lite / 全量 ----
    g_lite = parser.add_argument_group("训练规模（全量 / Lite）")
    g_scale = g_lite.add_mutually_exclusive_group()
    g_scale.add_argument(
        "--lite",
        action="store_true",
        help="跳过菜单，直接启用 Lite（约 lite_ratio 比例样本）",
    )
    g_scale.add_argument(
        "--full",
        action="store_true",
        help="跳过菜单，直接使用全量 COCOMask",
    )
    g_lite.add_argument(
        "--no_interactive",
        action="store_true",
        help="禁用启动菜单；未指定 --lite/--full 时默认全量",
    )
    g_lite.add_argument(
        "--lite_ratio",
        type=float,
        default=0.15,
        help="Lite 抽样比例，建议 0.10~0.20 (默认: 0.15)",
    )
    g_lite.add_argument(
        "--lite_seed",
        type=int,
        default=42,
        help="Lite 抽样随机种子 (默认: 42)",
    )
    g_lite.add_argument(
        "--lite_auto_tune",
        action="store_true",
        default=True,
        help="Lite 时自动下调 batch_size / num_epoch（默认开启）",
    )
    g_lite.add_argument(
        "--no_lite_auto_tune",
        action="store_false",
        dest="lite_auto_tune",
        help="禁用 Lite 自动下调超参",
    )

    # ---- 测试 ----
    g_test = parser.add_argument_group("测试")
    g_test.add_argument(
        "--test_iters", type=int, default=99, help="测试步（保留兼容，默认: 99）"
    )

    # ---- 输出 / CLI 体验 ----
    g_out = parser.add_argument_group("输出与进度")
    g_out.add_argument(
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
    g_out.add_argument(
        "--log_run",
        type=str,
        default=None,
        help="配合 --mode log_query：直接调出该 run_id（或前缀）",
    )
    g_out.add_argument(
        "--num_workers", type=int, default=1, help="DataLoader workers (默认: 1)"
    )
    g_out.add_argument(
        "--log_step", type=int, default=40, help="文本日志间隔（无进度条时）(默认: 40)"
    )
    g_out.add_argument(
        "--model_save_step", type=int, default=1, help="每隔多少 epoch 存盘 (默认: 1)"
    )
    g_out.add_argument(
        "--quiet", action="store_true", help="减少控制台啰嗦输出，保留关键进度与结果"
    )
    g_out.add_argument(
        "--verbose", action="store_true", help="打印更详细的中间信息"
    )
    g_out.add_argument(
        "--no_progress",
        action="store_true",
        help="关闭 tqdm 进度条（适合 nohup / 重定向）",
    )
    g_out.add_argument(
        "--save_viz",
        action="store_true",
        default=True,
        help="训练结束每个 epoch 保存可视化拼图 (默认开启)",
    )
    g_out.add_argument(
        "--no_save_viz",
        action="store_false",
        dest="save_viz",
        help="关闭 epoch 可视化拼图导出",
    )
    g_out.add_argument(
        "--no_epoch_plot",
        action="store_true",
        help="关闭每 epoch 结束后的 msg/den/loss 曲线弹窗",
    )
    g_out.add_argument(
        "--no_train_editor",
        action="store_true",
        help="跳过训练前超参编辑窗（仍可用命令行参数）",
    )

    # ---- 路径 ----
    g_dir = parser.add_argument_group("路径")
    g_dir.add_argument(
        "--image_dir",
        type=str,
        default="Datasets/COCOMask/train/train_class/",
        help="训练图像目录；嵌入默认改为 Datasets/images/（可用本参数覆盖）",
    )
    g_dir.add_argument(
        "--image_val_dir",
        type=str,
        default="Datasets/COCOMask/val/val_class/",
        help="验证目录；测准默认改为 Datasets/Recover/capture/",
    )
    g_dir.add_argument(
        "--psnr_ref_dir",
        type=str,
        default="Datasets/images_watermarked/",
        help="拍屏测准 PSNR 参考目录（同索引含水印宿主，默认 images_watermarked/）",
    )
    g_dir.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    g_dir.add_argument(
        "--model_save_dir", type=str, default="models", help="模型权重目录"
    )
    g_dir.add_argument(
        "--model_name",
        type=str,
        default="Encoder_Decoder_Model",
        help="权重文件名前缀",
    )
    g_dir.add_argument("--result_dir", type=str, default="results", help="结果输出目录")
    g_dir.add_argument(
        "--wmat_dir",
        type=str,
        default="results/WatermarkMatrix/w.mat",
        help="水印矩阵 .mat 路径",
    )

    return parser


# 训练默认宿主目录（argparse 默认）；嵌入/测准未显式指定时改用下方路径
_DEFAULT_TRAIN_IMAGE_DIR = "Datasets/COCOMask/train/train_class/"
_DEFAULT_EMBED_IMAGE_DIR = "Datasets/images/"
_DEFAULT_ACCURACY_IMAGE_DIR = "Datasets/Recover/capture/"

# 评估时与权重匹配的默认嵌入强度（用户显式 --embed_strength 时不覆盖）
_EVAL_STRENGTH_OFFICIAL = 0.0  # mask_99 / 论文整图
_EVAL_STRENGTH_TRAIN_DEFAULT = 0.0  # 与当前 --embed_strength 训练默认一致


def apply_mode_image_dirs(config: argparse.Namespace) -> argparse.Namespace:
    """
    按运行模式校正图像目录。

    - ``test_embedding`` → 默认 ``Datasets/images/``（编号宿主，非 COCOMask 拼接图）
    - ``test_accuracy`` → 默认 ``Datasets/Recover/capture/``
    - 命令行已传 ``--image_dir`` / ``--image_val_dir`` 时不覆盖
    """
    if config.mode == "test_embedding":
        if "--image_dir" not in sys.argv:
            config.image_dir = _DEFAULT_EMBED_IMAGE_DIR
            log_ok(f"嵌入宿主目录 → {config.image_dir}")
        if "--image_val_dir" not in sys.argv:
            # 嵌入不读验证集，与宿主目录对齐，避免误扫 COCOMask
            config.image_val_dir = config.image_dir
        # 粗检：若仍指向 COCOMask 拼接目录则警告
        tip = config.image_dir.replace("\\", "/").lower()
        if "cocomask" in tip:
            log_warn(
                "当前 image_dir 像是 COCOMask 拼接图目录；"
                "嵌入应使用左半边裁好的编号图（如 Datasets/images/）。"
                "可运行: python export_coco_hosts.py"
            )
    elif config.mode == "test_accuracy":
        if "--image_dir" not in sys.argv:
            config.image_dir = _DEFAULT_ACCURACY_IMAGE_DIR
            log_ok(f"测准图像目录 → {config.image_dir}")
        if "--image_val_dir" not in sys.argv:
            config.image_val_dir = config.image_dir
    return config


def apply_eval_embed_strength(config: argparse.Namespace) -> argparse.Namespace:
    """
    按权重选择嵌入方式，避免 strength 错配导致 PSNR/Acc 虚低。

    适用于 ``eval_mask`` / ``test_embedding`` / ``test_accuracy``。

    - 已设 ``weight_path`` → 读该文件旁 ``*.meta.txt``，默认 0
    - mask_99 → embed_strength=0
    - ss_best / best → 优先读对应 ``*.meta.txt``，否则 0
    - 命令行已传 ``--embed_strength`` → 不改
    """
    if getattr(config, "mode", None) not in (
        "eval_mask",
        "test_embedding",
        "test_accuracy",
    ):
        return config
    if "--embed_strength" in sys.argv:
        log_info(f"使用命令行 embed_strength={config.embed_strength}")
        return config

    wpath = getattr(config, "weight_path", None)
    if wpath:
        from pathlib import Path

        p = Path(wpath)
        strength = read_embed_strength_meta(p, default=0.0)
        config.embed_strength = float(strength)
        config.eval_ckpt = "custom"
        log_ok(
            f"weight_path → embed_strength={config.embed_strength} "
            f"（meta ← {p.name}）"
        )
        return config

    ckpt = normalize_ckpt_tag(getattr(config, "eval_ckpt", "99"))
    if ckpt == "custom":
        config.embed_strength = float(getattr(config, "embed_strength", 0.0) or 0.0)
        return config
    config.eval_ckpt = ckpt
    if ckpt in ("ss_best", "best"):
        strength = float(_EVAL_STRENGTH_TRAIN_DEFAULT)
        try:
            resolved = resolve_ckpt_path(
                config.model_save_dir,
                config.model_name,
                ckpt,
                distortion=getattr(config, "distortion", None),
            )
            strength = read_embed_strength_meta(resolved, default=strength)
            log_info(
                f"从 meta 读取 embed_strength={strength} ← {resolved}"
            )
        except FileNotFoundError:
            log_warn(
                f"尚未找到 {ckpt_label(ckpt)} 权重文件，embed_strength 暂用 {strength}"
            )
        config.embed_strength = strength
        log_ok(
            f"eval_ckpt={ckpt} ({ckpt_short(ckpt)}) → "
            f"embed_strength={config.embed_strength}"
        )
    else:
        config.embed_strength = float(_EVAL_STRENGTH_OFFICIAL)
        log_ok(
            f"eval_ckpt={ckpt} → embed_strength={config.embed_strength} "
            f"（官方整图输出）"
        )
    return config


def run_tool_mode(mode: str, extra_args: list[str] | None = None) -> int:
    """分发工具 / 实验脚本（不经过 Solver）。"""
    mapping: dict[str, tuple[Path, list[str]]] = {
        "tool_export_hosts": (REPO_ROOT / "tools" / "export_coco_hosts.py", []),
        "tool_verify_psnr": (REPO_ROOT / "tools" / "verify_host_psnr.py", []),
        "tool_verify_ssim": (REPO_ROOT / "tools" / "verify_host_ssim.py", []),
        "tool_verify_both": (REPO_ROOT / "tools" / "verify_host_embed.py", ["--metric", "both"]),
        "tool_crop_panels": (REPO_ROOT / "tools" / "crop_embed_panels.py", []),
        "tool_batch_rectify": (REPO_ROOT / "experiment" / "batch_rectify.py", []),
        "tool_rectify_gui": (REPO_ROOT / "experiment" / "rectify_gui.py", []),
        "tool_server_gui": (REPO_ROOT / "experiment" / "server_gui.py", []),
    }
    if mode not in mapping:
        log_err(f"未知工具模式: {mode}")
        return 2
    script, preset = mapping[mode]
    if not script.is_file():
        log_err(f"找不到脚本: {script}")
        return 1

    extra = list(extra_args or [])
    verify_modes = {"tool_verify_psnr", "tool_verify_ssim", "tool_verify_both"}
    # 交互：弹窗选 1~2 个 pth；已带 --ckpt / --no_interactive 则跳过
    if (
        mode in verify_modes
        and "--ckpt" not in extra
        and "--ckpt99" not in extra
        and "--no_interactive" not in sys.argv
        and "--ckpt" not in sys.argv
    ):
        from verify_weights_ui import ckpt_args_from_paths, prompt_embed_verify_weights

        titles = {
            "tool_verify_psnr": "嵌入 PSNR — 选择权重",
            "tool_verify_ssim": "嵌入 SSIM — 选择权重",
            "tool_verify_both": "嵌入质量 (PSNR+SSIM) — 选择权重",
        }
        picked = prompt_embed_verify_weights(
            title=titles.get(mode, "选择嵌入质量测试权重"),
            repo_root=REPO_ROOT,
        )
        if picked is None:
            log_warn("已取消权重选择，未启动测试。")
            return 130
        extra.extend(ckpt_args_from_paths(picked))
        log_ok(
            "已选权重: "
            + " | ".join(str(p) for p in picked)
            + ("（对照）" if len(picked) == 2 else "（单测）")
        )

    cmd = [sys.executable, str(script), *preset, *extra]
    cwd = str(script.parent) if script.parent.name == "experiment" else str(REPO_ROOT)
    print_mode_badge(mode)
    log_info(f"启动: {' '.join(cmd)}")
    log_info(f"工作目录: {cwd}")
    try:
        completed = subprocess.run(cmd, cwd=cwd, check=False)
    except KeyboardInterrupt:
        log_warn("已中断工具进程")
        return 130
    code = int(completed.returncode)
    if code == 0:
        log_ok("工具执行完成。")
    else:
        log_err(f"工具退出码: {code}")
    return code


def resolve_run_mode(config: argparse.Namespace) -> argparse.Namespace:
    """
    决定运行模式。

    优先级
    ------
    1. 命令行已给 ``--mode`` → 直接使用
    2. ``--no_interactive`` → 默认 ``eval_mask``（跳过训练评估）
    3. 交互功能面板：训练评估 / 工具 / 实验
    """
    if config.mode is not None:
        if (
            config.mode == "quit"
            or config.mode == "log_query"
            or str(config.mode).startswith("tool_")
        ):
            return config
        if config.dataset == "train_mask" and config.mode != "train_mask":
            # 若用户只改了 mode，同步 dataset
            if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
                config.dataset = config.mode
        if config.distortion is None:
            if config.mode == "train_mask":
                if config.no_interactive:
                    config.distortion = "ScreenShootingMB"
                    log_info("未指定 --distortion → 默认 ScreenShootingMB (DMB)")
                else:
                    config.distortion = prompt_noise_layer(default_choice="2")
            elif config.mode not in (
                "eval_mask",
                "test_embedding",
                "test_accuracy",
            ):
                config.distortion = "ScreenShooting"
        # 显式 --mode：未指定权重时弹窗选 .pth（或 --no_interactive 用默认标签）
        if (
            config.mode in ("eval_mask", "test_embedding", "test_accuracy")
            and not getattr(config, "weight_path", None)
            and "--weight_path" not in sys.argv
            and config.eval_ckpt is None
            and not config.no_interactive
            and "--eval_ckpt" not in sys.argv
        ):
            from verify_weights_ui import (
                infer_noise_menu_choice_from_path,
                prompt_single_weight_file,
            )

            titles = {
                "eval_mask": "评估模式 — 选择权重",
                "test_embedding": "水印叠加 — 选择权重",
                "test_accuracy": "拍屏测准 — 选择权重",
            }
            picked = prompt_single_weight_file(
                title=titles.get(config.mode, "选择权重"),
                repo_root=REPO_ROOT,
            )
            if picked is None:
                log_warn("已取消权重选择。")
                config.mode = "quit"
                return config
            config.weight_path = str(picked)
            config.eval_ckpt = "custom"
            if config.embedding_epoch == 0:
                config.embedding_epoch = 99
            log_ok(f"权重文件：{picked}")
            if config.distortion is None and "--distortion" not in sys.argv:
                config.distortion = prompt_noise_layer(
                    default_choice=infer_noise_menu_choice_from_path(picked)
                )
        elif (
            config.mode in ("eval_mask", "test_embedding", "test_accuracy")
            and getattr(config, "weight_path", None)
        ):
            config.eval_ckpt = "custom"
            if config.embedding_epoch == 0:
                config.embedding_epoch = 99
        elif (
            config.mode in ("eval_mask", "test_embedding", "test_accuracy")
            and config.eval_ckpt is None
        ):
            config.eval_ckpt = "99"
            if config.embedding_epoch == 0:
                config.embedding_epoch = 99
        elif config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
            if str(config.eval_ckpt) != "custom":
                config.eval_ckpt = normalize_ckpt_tag(config.eval_ckpt)
            if config.eval_ckpt == "99" and config.embedding_epoch == 0:
                config.embedding_epoch = 99
        if (
            config.mode in ("eval_mask", "test_embedding", "test_accuracy")
            and config.distortion is None
        ):
            if getattr(config, "weight_path", None):
                from verify_weights_ui import infer_noise_menu_choice_from_path

                if config.no_interactive:
                    choice = infer_noise_menu_choice_from_path(config.weight_path)
                    config.distortion = (
                        "ScreenShootingMB" if choice == "2" else "ScreenShooting"
                    )
                    log_info(f"未指定 --distortion → 默认 {config.distortion}")
                else:
                    config.distortion = prompt_noise_layer(
                        default_choice=infer_noise_menu_choice_from_path(
                            config.weight_path
                        )
                    )
            elif config.no_interactive:
                config.distortion = ckpt_default_distortion(config.eval_ckpt)
                log_info(f"未指定 --distortion → 默认 {config.distortion}")
            else:
                config.distortion = prompt_noise_layer(
                    default_choice=default_noise_menu_choice(config.eval_ckpt)
                )
        if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
            apply_eval_embed_strength(config)
        # 显式 --mode train_mask 且未指定初始化方式时，交互询问
        if (
            config.mode == "train_mask"
            and getattr(config, "init_from_ckpt", None) is None
            and config.init_from_epoch is None
            and config.embedding_epoch == 0
            and not config.no_interactive
            and "--init_from_epoch" not in sys.argv
            and "--init_from_ckpt" not in sys.argv
        ):
            init_cfg = prompt_train_init(default_choice="1")
            config.init_from_ckpt = init_cfg.get("init_from_ckpt")
            config.init_from_epoch = init_cfg.get("init_from_epoch")
            if "embedding_epoch" in init_cfg:
                config.embedding_epoch = int(init_cfg["embedding_epoch"])
        elif config.mode == "train_mask" and getattr(config, "init_from_ckpt", None):
            config.init_from_ckpt = normalize_ckpt_tag(config.init_from_ckpt)
        return config

    if config.no_interactive:
        config.mode = "eval_mask"
        config.dataset = "eval_mask"
        if config.eval_ckpt is None:
            config.eval_ckpt = "99"
        if config.embedding_epoch == 0:
            config.embedding_epoch = 99
        if config.distortion is None:
            config.distortion = "ScreenShooting"
        apply_eval_embed_strength(config)
        log_info(
            f"--no_interactive 且未指定 --mode → 默认 eval_mask "
            f"(ckpt={config.eval_ckpt}, strength={config.embed_strength})"
        )
        return config

    chosen = prompt_run_mode(default_choice="0")
    config.mode = chosen["mode"]
    if (
        config.mode == "quit"
        or config.mode == "log_query"
        or str(config.mode).startswith("tool_")
    ):
        return config
    config.dataset = chosen["dataset"]
    if "distortion" in chosen:
        config.distortion = chosen["distortion"]
    elif config.distortion is None:
        config.distortion = "ScreenShooting"
    if "init_from_epoch" in chosen:
        config.init_from_epoch = chosen["init_from_epoch"]
    if "init_from_ckpt" in chosen:
        config.init_from_ckpt = chosen["init_from_ckpt"]
    if "lite" in chosen:
        # 菜单已选定 lite 时，避免后续再弹训练规模菜单
        config.lite = bool(chosen["lite"])
        if chosen["mode"] in ("train_mask", "eval_mask"):
            config._scale_from_menu = True
    if "eval_ckpt" in chosen:
        config.eval_ckpt = chosen["eval_ckpt"]
    elif config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
        if config.eval_ckpt is None and not getattr(config, "weight_path", None):
            config.eval_ckpt = "99"
    if "weight_path" in chosen:
        config.weight_path = chosen["weight_path"]
        config.eval_ckpt = "custom"
    if "embedding_epoch" in chosen:
        config.embedding_epoch = int(chosen["embedding_epoch"])
    if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
        # 若菜单返回了明确 strength（如 99→0），先写入再统一校正
        if (
            "embed_strength" in chosen
            and chosen["embed_strength"] is not None
            and "--embed_strength" not in sys.argv
        ):
            config.embed_strength = float(chosen["embed_strength"])
        apply_eval_embed_strength(config)
    return config


def resolve_train_scale(config: argparse.Namespace) -> argparse.Namespace:
    """
    决定全量 / Lite。

    优先级
    ------
    1. 命令行 ``--lite`` / ``--full``（跳过菜单）
    2. 功能菜单已选定规模（``_scale_from_menu``）
    3. ``--no_interactive`` → 默认全量（评估除外：沿用菜单/参数）
    4. 训练相关模式 → 交互询问（1=全量，2=Lite）
    5. 测试 / 评估模式 → 默认全量（不弹菜单），除非菜单已设 lite
    """
    if config.lite and config.full:
        log_err("--lite 与 --full 不能同时使用")
        sys.exit(2)

    if config.lite:
        log_info("命令行指定 --lite，跳过规模菜单")
        return config

    if config.full:
        config.lite = False
        log_info("命令行指定 --full，跳过规模菜单 → 全量模式")
        return config

    if getattr(config, "_scale_from_menu", False):
        log_info(
            "规模已由功能面板决定 → "
            + ("Lite" if config.lite else "全量")
        )
        return config

    if config.no_interactive:
        if config.mode != "eval_mask":
            config.lite = False
            log_info("--no_interactive：未指定规模，默认全量模式")
        return config

    # 仅训练时询问；嵌入/测准/评估通常不需要（评估可在功能菜单里选 Lite）
    if config.mode == "train_mask":
        config.lite = prompt_train_scale(default_choice="2")
    else:
        if not hasattr(config, "lite") or config.mode != "eval_mask":
            # eval 若未在菜单设 lite，保持 False
            pass
        if not config.quiet:
            log_info(f"当前 mode={config.mode}，跳过训练规模菜单")

    return config


def apply_scale_defaults(config: argparse.Namespace) -> argparse.Namespace:
    """
    按全量 / Lite 设置未显式指定的规模相关默认值。

    - warmup_epochs：全量默认 1，Lite 默认 5（命令行显式传入则不改）
    """
    if config.warmup_epochs is None:
        config.warmup_epochs = 5 if config.lite else 0
        log_info(
            f"warmup_epochs → {config.warmup_epochs} "
            f"({'Lite' if config.lite else '全量'} 默认)"
        )
    return config


def apply_lite_defaults(config: argparse.Namespace) -> argparse.Namespace:
    """
    Lite 模式下自动下调 / 对齐适合小批量的超参。

    - batch_size > 8 时降为 8
    - num_epoch 对齐到约 40（给 ScreenShooting 足够时间）
    """
    if not config.lite:
        return config

    # 评估 / 嵌入 / 测准：只抽样数据，不改训练超参
    if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
        return config

    if config.lite_ratio < 0.10 or config.lite_ratio > 0.20:
        log_warn(
            f"lite_ratio={config.lite_ratio} 超出推荐区间 [0.10, 0.20]，"
            "仍将按给定值抽样"
        )

    if config.lite_auto_tune:
        if config.batch_size > 4:
            log_info(f"Lite auto-tune: batch_size {config.batch_size} → 8")
            config.batch_size = 4
        if config.num_epoch < 40:
            log_info(f"Lite auto-tune: num_epoch {config.num_epoch} → 40")
            config.num_epoch = 40
        elif config.num_epoch > 60:
            log_info(f"Lite auto-tune: num_epoch {config.num_epoch} → 40")
            config.num_epoch = 40
        # 保证 warmup 不会吃掉全部 epoch
        if config.warmup_epochs >= config.num_epoch:
            config.warmup_epochs = max(1, config.num_epoch // 5)
            log_info(f"Lite auto-tune: warmup_epochs → {config.warmup_epochs}")

    return config


def _ensure_dirs(config: argparse.Namespace) -> None:
    """创建日志与模型保存目录。"""
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)


def main(config: argparse.Namespace) -> None:
    """组装 DataLoader 与 Solver，按 mode 分发。"""
    try:
        import torch
        from torch.backends import cudnn
        from data_loader import get_loader
        from solver import Solver
    except ImportError as e:
        log_err(f"缺少依赖: {e}")
        log_err("请先执行: pip install -r requirements.txt")
        sys.exit(1)

    cudnn.benchmark = True
    _ensure_dirs(config)

    show_progress = not config.no_progress
    config.show_progress = show_progress  # 透传给 Solver

    # eval / embed / accuracy：统一 ckpt 默认与 strength
    if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
        if config.mode == "eval_mask":
            config.dataset = "eval_mask"
            config.image_dir = config.image_val_dir
        if getattr(config, "weight_path", None):
            config.eval_ckpt = "custom"
            if config.embedding_epoch == 0:
                config.embedding_epoch = 99
            log_info(f"{config.mode}：使用自定义权重 {config.weight_path}")
        else:
            if getattr(config, "eval_ckpt", None) is None:
                config.eval_ckpt = "99"
            if str(config.eval_ckpt) != "custom":
                config.eval_ckpt = normalize_ckpt_tag(config.eval_ckpt)
            if config.eval_ckpt == "99" and config.embedding_epoch == 0:
                config.embedding_epoch = 99
                log_info(f"{config.mode}：未指定 epoch → 使用 mask_99")
            elif config.eval_ckpt in ("ss_best", "best"):
                log_info(
                    f"{config.mode}：使用 {ckpt_label(config.eval_ckpt)} "
                    f"({named_best_filename(config.model_name, config.eval_ckpt)})"
                )
        apply_eval_embed_strength(config)

    apply_mode_image_dirs(config)

    # ---- DataLoader ----
    common_kw = dict(
        image_size=config.image_size,
        batch_size=config.batch_size,
        dataset=config.dataset,
        mode=config.mode,
        num_workers=config.num_workers,
        w_path=config.wmat_dir,
        lite=config.lite,
        lite_ratio=config.lite_ratio,
        lite_seed=config.lite_seed,
    )

    try:
        data_loader = get_loader(config.image_dir, **common_kw)
        data_loader_test = get_loader(config.image_val_dir, **common_kw)
    except FileNotFoundError as e:
        log_err(str(e))
        if config.mode == "test_embedding":
            log_err(
                "嵌入请将编号宿主图放到 Datasets/images/ "
                "（或传 --image_dir）。可用: python tools/export_coco_hosts.py "
                "或主菜单选项 5"
            )
        elif config.mode == "test_accuracy":
            log_err("测准请确认 --image_dir 指向矫正后的拍屏图目录")
        else:
            log_err("请确认 --image_dir / --image_val_dir 指向有效目录")
        sys.exit(1)

    # ---- 配置摘要 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.mode in ("eval_mask", "test_embedding", "test_accuracy"):
        if getattr(config, "weight_path", None):
            weight_path = str(config.weight_path)
        else:
            try:
                weight_path = resolve_ckpt_path(
                    config.model_save_dir,
                    config.model_name,
                    normalize_ckpt_tag(getattr(config, "eval_ckpt", "99")),
                    embedding_epoch=int(config.embedding_epoch or 99),
                    distortion=config.distortion,
                )
            except FileNotFoundError as e:
                weight_path = str(e).split("\n")[0]
    else:
        weight_path = os.path.join(
            config.model_save_dir,
            config.distortion,
            f"{config.model_name}_mask_{config.embedding_epoch}.pth",
        )
    extra = {
        "device": device,
        "train_samples": len(data_loader.dataset),
        "val_samples": len(data_loader_test.dataset),
        "show_progress": show_progress,
        "weight_path": weight_path,
        "eval_ckpt": getattr(config, "eval_ckpt", None),
        "init_from_ckpt": getattr(config, "init_from_ckpt", None),
    }
    if device == "cuda":
        try:
            extra["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    if not config.quiet:
        print_mode_badge(config.mode)
        print_config_summary(config, extra=extra)
        if device != "cuda":
            log_warn("当前为 CPU 训练。若本机有 NVIDIA，请用 CUDA 版 PyTorch 环境启动。")

    # ---- Solver ----
    solver = Solver(data_loader, data_loader_test, config)

    def _weight_hint() -> str:
        if getattr(config, "weight_path", None):
            return str(config.weight_path)
        try:
            return resolve_ckpt_path(
                config.model_save_dir,
                config.model_name,
                normalize_ckpt_tag(getattr(config, "eval_ckpt", "99")),
                embedding_epoch=int(config.embedding_epoch or 99),
                distortion=config.distortion,
            )
        except FileNotFoundError:
            return named_best_filename(
                config.model_name, normalize_ckpt_tag(getattr(config, "eval_ckpt", "99"))
            )

    if config.mode == "train_mask":
        log_ok("开始训练 (train_mask)" + (" [LITE]" if config.lite else ""))
        solver.train_mask()
    elif config.mode == "eval_mask":
        log_ok(f"跳过训练，直接评估 (eval_mask) | {_weight_hint()}")
        solver.eval_mask()
    elif config.mode == "test_accuracy":
        log_ok(f"开始准确率测试 (test_accuracy) | {_weight_hint()}")
        solver.test_accuracy()
    elif config.mode == "test_embedding":
        log_ok(f"开始水印嵌入 (test_embedding) | {_weight_hint()}")
        solver.test_embedding()
    else:
        log_err(f"未知 mode: {config.mode}")
        sys.exit(1)

    log_ok("全部完成。")


if __name__ == "__main__":
    # Windows 终端默认 GBK，强制 UTF-8 避免中文帮助/横幅乱码
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = build_parser()
    config = parser.parse_args()
    print_banner(lite=False)
    config = resolve_run_mode(config)

    if config.mode == "quit":
        log_ok("已退出。")
        raise SystemExit(0)

    if config.mode == "log_query":
        print_mode_badge("log_query")
        if getattr(config, "log_run", None):
            from train_log import format_run_detail, load_run

            try:
                doc = load_run(config.log_dir, config.log_run)
            except FileNotFoundError as e:
                log_err(str(e))
                raise SystemExit(1)
            print(format_run_detail(doc), flush=True)
            log_ok(f"JSON: {doc.get('json_path')}")
            raise SystemExit(0)
        raise SystemExit(prompt_log_browser(getattr(config, "log_dir", "logs")))

    if str(config.mode).startswith("tool_"):
        raise SystemExit(run_tool_mode(config.mode))

    config = resolve_train_scale(config)
    if config.lite:
        print_lite_badge()
    config = apply_scale_defaults(config)
    config = apply_lite_defaults(config)

    if config.mode == "train_mask":
        from train_ui import edit_train_hparams, print_train_hparams

        # 基本菜单选完后 → 弹窗改超参 → 控制台打印明细 → 开训
        config = edit_train_hparams(config)
        print_train_hparams(config)

    main(config)
