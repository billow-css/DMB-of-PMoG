#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solver.py
=========
PIMoG 训练 / 嵌入 / 测准调度器。

职责
----
- 构建 Encoder_Decoder + Discriminator，管理优化器与设备
- train_mask()      ：对抗训练 + 感知 mask 损失，带实时 CLI 进度
- eval_mask()       ：跳过训练，加载已有权重做验证集 Acc/BER/PSNR
- test_embedding()  ：嵌入水印并导出五联可视化拼图
- test_accuracy()   ：比特级准确率评估 + 相对含水印宿主的 PSNR

损失组成（训练，与官方仓库一致）
--------------------------------
L = λ1 * L_message + λ2 * L_denoise + λ3 * L_GAN
L_denoise = 0.5 * MSE(E⊙∇mask, x⊙∇mask) + 2 * MSE(E⊙v_mask, x⊙v_mask)

指标（与官方一致）
----------------
Acc = 1 - BER
BER = mean(|bit(m̂) - m|)，bit(.) 使用阈值 0.5

公开接口
--------
- Solver(data_loader, data_loader_test, config)
"""

from __future__ import annotations

import os
import shutil
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from cli_utils import (
    ProgressMonitor,
    format_eta,
    log_info,
    log_ok,
    log_step,
    log_warn,
    wrap_iterable,
)
from model import Discriminator, Encoder_Decoder
from Noise_Layer import Identity, ScreenShooting, ScreenShootingMB
from ckpt_utils import (
    ckpt_label,
    ckpt_run_tag,
    ckpt_short,
    named_best_filename,
    normalize_ckpt_tag,
    resolve_ckpt_path,
)
from ui.train_log import TrainRunLogger
from ui.train_ui import show_epoch_curves


class Solver(object):
    """PIMoG 训练与测试调度器。"""

    def __init__(self, data_loader, data_loader_test, config):
        """
        Parameters
        ----------
        data_loader : DataLoader
            训练集（或嵌入集）。
        data_loader_test : DataLoader
            验证 / 测试集。
        config : argparse.Namespace
            命令行配置（需含 image_size, lambda*, distortion 等字段）。
        """
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test

        self.image_size = config.image_size
        self.num_channels = config.num_channels
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.lambda1 = config.lambda1
        self.lambda2 = config.lambda2
        self.lambda3 = config.lambda3
        self.num_epoch = config.num_epoch
        self.distortion = config.distortion
        self.test_iters = config.test_iters
        self.lr = float(getattr(config, "lr", 1e-3))
        self.warmup_epochs = int(getattr(config, "warmup_epochs", 0))
        self.embed_strength = float(getattr(config, "embed_strength", 0.0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            try:
                torch.cuda.set_device(self.device)
            except Exception:
                pass
        self._log_device_banner()

        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir
        self.model_name = config.model_name
        self.result_dir = config.result_dir
        self.embedding_epoch = config.embedding_epoch
        self.init_from_epoch = getattr(config, "init_from_epoch", None)
        self.init_from_ckpt = getattr(config, "init_from_ckpt", None)
        if self.init_from_ckpt:
            self.init_from_ckpt = normalize_ckpt_tag(self.init_from_ckpt)
        elif self.init_from_epoch is not None and int(self.init_from_epoch) == 99:
            self.init_from_ckpt = "99"
        self.eval_ckpt = normalize_ckpt_tag(
            str(getattr(config, "eval_ckpt", "99") or "99")
        )
        self.eval_weight_path = getattr(config, "weight_path", None) or getattr(
            config, "eval_weight_path", None
        )
        if self.eval_weight_path:
            self.eval_weight_path = str(self.eval_weight_path)
            self.eval_ckpt = "custom"

        self.log_step = config.log_step
        self.model_save_step = config.model_save_step

        # CLI / 输出选项（带默认值，兼容旧调用）
        self.show_progress = getattr(config, "show_progress", True)
        self.quiet = getattr(config, "quiet", False)
        self.verbose = getattr(config, "verbose", False)
        self.save_viz = getattr(config, "save_viz", True)
        self.lite = getattr(config, "lite", False)
        self.psnr_ref_dir = str(
            getattr(config, "psnr_ref_dir", None) or "Datasets/images_watermarked/"
        )
        self.no_epoch_plot = bool(getattr(config, "no_epoch_plot", False))
        self.epoch_plot_block = bool(getattr(config, "epoch_plot_block", True))

        self.build_model()

    def _log_device_banner(self) -> None:
        """启动时明确打印算力设备，避免把核显 3D（曲线窗）误当成训练设备。"""
        if self.device.type == "cuda" and torch.cuda.is_available():
            idx = self.device.index if self.device.index is not None else 0
            name = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            log_ok(
                f"训练算力设备: {self.device} | {name} | "
                f"CC {cap[0]}.{cap[1]} | VRAM {mem_gb:.1f} GB"
            )
            log_info(
                "任务管理器：请看 NVIDIA 卡片的「CUDA」或「Compute_0」引擎，"
                "不要只看「3D」（纯算力时 3D 常接近 0%）。"
            )
            log_info(
                "Intel 核显「3D」若在 2%~30% 浮动，多半是 epoch 曲线弹窗 / 桌面合成，"
                "不是训练主算力。可用 --no_epoch_plot 关闭弹窗对照。"
            )
        else:
            log_warn(
                "训练算力设备: CPU（torch.cuda.is_available()=False）。"
                "请确认当前解释器是带 CUDA 的 PyTorch，例如 "
                "D:\\anaconda\\envs\\pytorch2.4.0\\python.exe"
            )

    def _ed_module(self):
        """取 Encoder_Decoder 本体（兼容 DataParallel）。"""
        return self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net

    def _set_noiser(self, kind: str) -> None:
        """切换噪声层：Identity / ScreenShooting / ScreenShootingMB。"""
        mod = self._ed_module()
        if kind == "Identity":
            mod.Noiser = Identity()
            mod.distortion = "Identity"
        elif kind == "ScreenShooting":
            mod.Noiser = ScreenShooting()
            mod.distortion = "ScreenShooting"
        elif kind == "ScreenShootingMB":
            mod.Noiser = ScreenShootingMB()
            mod.distortion = "ScreenShootingMB"
        else:
            raise ValueError(kind)
        log_info(f"噪声层切换 → {kind}")

    def _find_weight_file(self, epoch: int) -> str | None:
        """在各噪声层目录中查找 ``…_mask_{epoch}.pth``，找到则返回路径。"""
        fname = f"{self.model_name}_mask_{epoch}.pth"
        candidates = [
            os.path.join(self.model_save_dir, self.distortion, fname),
            os.path.join(self.model_save_dir, "ScreenShooting", fname),
            os.path.join(self.model_save_dir, "ScreenShootingMB", fname),
            os.path.join(self.model_save_dir, fname),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _copy_pretrained_init(self, epoch: int) -> str:
        """
        将预训练 ``mask_{epoch}`` 拷贝到当前 distortion 目录，供本次训练挂载。

        目标文件名：``{model_name}_mask_init{epoch}.pth``（避免覆盖正在写的 epoch 权重）。
        """
        src = self._find_weight_file(epoch)
        if src is None:
            raise FileNotFoundError(
                f"找不到预训练权重 mask_{epoch}.pth，已搜索 models/ 下各噪声层目录"
            )
        dst_dir = os.path.join(self.model_save_dir, self.distortion)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{self.model_name}_mask_init{epoch}.pth")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
            log_ok(f"已拷贝预训练权重:\n         {src}\n      →  {dst}")
        else:
            log_info(f"预训练权重已在目标位置: {dst}")
        return dst

    def _find_last_train_result_image(self) -> str | None:
        """查找训练可视化最后一轮：results/Image/images/ 中编号最大的 png。"""
        img_dir = os.path.join(self.result_dir, "Image", "images")
        if not os.path.isdir(img_dir):
            return None
        best_ep = -1
        best_path = None
        for name in os.listdir(img_dir):
            if not name.lower().endswith(".png"):
                continue
            stem = os.path.splitext(name)[0]
            try:
                ep = int(stem)
            except ValueError:
                continue
            if ep >= best_ep:
                best_ep = ep
                best_path = os.path.join(img_dir, name)
        return best_path

    def _draw_metrics_board(
        self,
        bit_acc: float,
        bit_ber: float,
        psnr: float,
        width: int = 640,
        height: int = 360,
    ) -> np.ndarray:
        """绘制评估指标看板（BGR uint8）。"""
        board = np.full((height, width, 3), 245, dtype=np.uint8)
        lines = [
            "PIMoG Eval Results",
            f"Noiser: {self._ed_module().distortion}",
            f"Weight epoch: {self.embedding_epoch}",
            f"ValAcc: {bit_acc:.2f}%",
            f"BER:    {bit_ber:.4f}",
            f"PSNR:   {psnr:.2f} dB",
            f"embed_strength: {self.embed_strength:g}",
        ]
        y = 48
        for i, text in enumerate(lines):
            scale = 1.1 if i == 0 else 0.85
            thickness = 2 if i == 0 else 1
            color = (40, 40, 40) if i != 0 else (20, 90, 180)
            cv2.putText(
                board,
                text,
                (36, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += 42 if i == 0 else 36
        return board

    def _hconcat_resize(self, images: list[np.ndarray], row_h: int = 256) -> np.ndarray:
        """将多张图缩放到同一高度后横向拼接。"""
        resized = []
        for im in images:
            if im is None:
                continue
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            h, w = im.shape[:2]
            nh = row_h
            nw = max(1, int(round(w * (nh / max(h, 1)))))
            resized.append(cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA))
        if not resized:
            return np.full((row_h, row_h, 3), 200, dtype=np.uint8)
        return np.concatenate(resized, axis=1)

    def _try_open_image(self, path: str) -> None:
        """尝试用系统默认程序打开图片（Windows startfile）。"""
        try:
            if os.name == "nt" and os.path.isfile(path):
                os.startfile(path)  # type: ignore[attr-defined]
                log_info(f"已打开结果图: {path}")
        except Exception as e:
            log_warn(f"无法自动打开图片: {e}")

    def _eval_run_tag(self) -> str:
        """评估结果子目录标签：ss_best / mb_best / ep{N} / 自定义权重文件名。"""
        if self.eval_weight_path:
            stem = os.path.splitext(os.path.basename(self.eval_weight_path))[0]
            safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
            return (safe or "custom")[:64]
        return ckpt_run_tag(self.eval_ckpt, embedding_epoch=int(self.embedding_epoch))

    def _resolve_eval_weight_path(self) -> str:
        """解析评估 / 嵌入 / 测准权重路径（自定义文件 或 99 / ss_best / best）。"""
        if self.eval_weight_path:
            if not os.path.isfile(self.eval_weight_path):
                raise FileNotFoundError(f"权重文件不存在: {self.eval_weight_path}")
            log_info(f"自定义权重 → {self.eval_weight_path}")
            return self.eval_weight_path
        try:
            path = resolve_ckpt_path(
                self.model_save_dir,
                self.model_name,
                self.eval_ckpt,
                embedding_epoch=int(self.embedding_epoch or 99),
                distortion=self.distortion,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e)) from e
        log_info(
            f"权重标签={self.eval_ckpt} ({ckpt_short(self.eval_ckpt)}) → {path}"
        )
        return path

    def _copy_pretrained_named(self, tag: str) -> str:
        """
        将命名预训练权重拷贝到当前 distortion 目录，供本次训练挂载。
        目标：``{model}_mask_init_{tag}.pth``（不覆盖正在写的 epoch 权重）。
        """
        tag = normalize_ckpt_tag(tag)
        src = resolve_ckpt_path(
            self.model_save_dir,
            self.model_name,
            tag,
            embedding_epoch=99 if tag == "99" else int(self.embedding_epoch or 99),
            distortion=self.distortion,
        )
        dst_dir = os.path.join(self.model_save_dir, self.distortion)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(
            dst_dir, f"{self.model_name}_mask_init_{tag}.pth"
        )
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
            log_ok(
                f"已拷贝预训练权重 ({ckpt_label(tag)}):\n"
                f"         {src}\n      →  {dst}"
            )
        else:
            log_info(f"预训练权重已在目标位置: {dst}")
        return dst

    # ------------------------------------------------------------------
    # 模型构建
    # ------------------------------------------------------------------
    def build_model(self) -> None:
        """按 dataset 类型构建网络并加载权重（如需要）。"""
        weight_dir = os.path.join(self.model_save_dir, self.distortion)
        weight_path = os.path.join(
            weight_dir,
            f"{self.model_name}_mask_{self.embedding_epoch}.pth",
        )

        if self.dataset in ("train_mask", "eval_mask"):
            self.net = Encoder_Decoder(
                self.distortion, embed_strength=self.embed_strength
            )
            self.print_network(self.net, self.dataset)
            self.net.to(self.device)
            self.net = torch.nn.DataParallel(self.net)

            if self.dataset == "eval_mask":
                # 仅评估：必须加载已有权重，不建优化器 / 判别器
                weight_path = self._resolve_eval_weight_path()
                log_info(f"加载评估权重: {weight_path}")
                state = torch.load(weight_path, map_location=self.device)
                # Noiser 无参；跨 ScreenShooting↔MB 加载时忽略无关键
                missing, unexpected = self.net.load_state_dict(state, strict=False)
                if missing:
                    log_warn(f"权重缺失键（可忽略若仅为 Noiser）: {len(missing)} 个")
                if unexpected:
                    log_warn(f"权重多余键: {len(unexpected)} 个")
                log_ok(
                    f"eval_mask 就绪 | Noiser={self.distortion} | "
                    f"embed_strength={self.embed_strength} | "
                    f"ckpt={ckpt_short(self.eval_ckpt)}"
                    + (
                        f" | file={os.path.basename(self.eval_weight_path)}"
                        if self.eval_weight_path
                        else (
                            f" | epoch={self.embedding_epoch}"
                            if self.eval_ckpt == "99"
                            else ""
                        )
                    )
                    + f"\n         weights={weight_path}"
                )
            else:
                self.net_Discriminator = Discriminator(self.num_channels)
                self.net_Discriminator.to(self.device)
                # 必须在 DataParallel 之后创建优化器，确保参数引用一致
                self.optimizer_Discriminator = optim.Adam(
                    self.net_Discriminator.parameters(), lr=self.lr
                )
                self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

                # 挂载预训练拷贝初始化（99 / ss_best / best）——训练 epoch 仍从 embedding_epoch 起计
                if self.init_from_ckpt:
                    init_path = self._copy_pretrained_named(self.init_from_ckpt)
                    log_info(f"加载预训练初始化权重: {init_path}")
                    state = torch.load(init_path, map_location=self.device)
                    missing, unexpected = self.net.load_state_dict(state, strict=False)
                    if missing:
                        log_warn(f"初始化缺失键: {len(missing)} 个")
                    if unexpected:
                        log_warn(f"初始化多余键: {len(unexpected)} 个")
                    log_ok(
                        f"已挂载预训练 {ckpt_label(self.init_from_ckpt)} 拷贝；"
                        f"本次训练从 epoch={self.embedding_epoch} 起记"
                    )
                elif self.init_from_epoch is not None and int(self.init_from_epoch) > 0:
                    init_path = self._copy_pretrained_init(int(self.init_from_epoch))
                    log_info(f"加载预训练初始化权重: {init_path}")
                    state = torch.load(init_path, map_location=self.device)
                    missing, unexpected = self.net.load_state_dict(state, strict=False)
                    if missing:
                        log_warn(f"初始化缺失键: {len(missing)} 个")
                    if unexpected:
                        log_warn(f"初始化多余键: {len(unexpected)} 个")
                    log_ok(
                        f"已挂载预训练 mask_{self.init_from_epoch} 拷贝；"
                        f"本次训练从 epoch={self.embedding_epoch} 起记"
                    )
                elif self.embedding_epoch != 0:
                    log_info(f"加载续训权重: {weight_path}")
                    self.net.load_state_dict(
                        torch.load(weight_path, map_location=self.device),
                        strict=False,
                    )

                # 强噪声 + 从头训练：Identity warmup（挂载任意预训练时跳过）
                do_warmup = (
                    self.distortion in ("ScreenShooting", "ScreenShootingMB")
                    and self.warmup_epochs > 0
                    and self.embedding_epoch == 0
                    and self.init_from_ckpt is None
                    and self.init_from_epoch is None
                )
                if do_warmup:
                    self._set_noiser("Identity")
                    log_ok(
                        f"Warmup {self.warmup_epochs} epoch：Identity → 随后 "
                        f"{self.distortion} | embed_strength={self.embed_strength}"
                    )
                else:
                    init_hint = ""
                    if self.init_from_ckpt:
                        init_hint = f" | init_from={ckpt_short(self.init_from_ckpt)}"
                    elif self.init_from_epoch is not None:
                        init_hint = f" | init_from={self.init_from_epoch}"
                    log_info(
                        f"优化器 lr={self.lr} | embed_strength={self.embed_strength} | "
                        f"λ=({self.lambda1},{self.lambda2},{self.lambda3}) | "
                        f"noise={self.distortion}{init_hint}"
                    )
        elif self.dataset in ("test_embedding",):
            self.net_ED = Encoder_Decoder(
                self.distortion, embed_strength=self.embed_strength
            )
            self.net_ED = self.net_ED.to(self.device)
            self.net_E = self.net_ED.Encoder
            self.net_ED = torch.nn.DataParallel(self.net_ED)
            weight_path = self._resolve_eval_weight_path()
            log_info(f"加载嵌入权重: {weight_path}")
            state = torch.load(weight_path, map_location=self.device)
            missing, unexpected = self.net_ED.load_state_dict(state, strict=False)
            if missing:
                log_warn(f"权重缺失键（可忽略若仅为 Noiser）: {len(missing)} 个")
            if unexpected:
                log_warn(f"权重多余键: {len(unexpected)} 个")
            log_ok(
                f"test_embedding 就绪 | Noiser={self.distortion} | "
                f"embed_strength={self.embed_strength} | "
                f"ckpt={ckpt_short(self.eval_ckpt)}"
                + (
                    f" | epoch={self.embedding_epoch}"
                    if self.eval_ckpt == "99"
                    else ""
                )
                + f"\n         weights={weight_path}"
            )

        elif self.dataset in ("test_accuracy",):
            self.net = Encoder_Decoder(
                self.distortion, embed_strength=self.embed_strength
            )
            self.print_network(self.net, self.dataset)
            self.net.to(self.device)
            self.net_D = self.net.Decoder
            self.net = torch.nn.DataParallel(self.net)
            weight_path = self._resolve_eval_weight_path()
            log_info(f"加载测准权重: {weight_path}")
            state = torch.load(weight_path, map_location=self.device)
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            if missing:
                log_warn(f"权重缺失键（可忽略若仅为 Noiser）: {len(missing)} 个")
            if unexpected:
                log_warn(f"权重多余键: {len(unexpected)} 个")
            log_ok(
                f"test_accuracy 就绪 | Noiser={self.distortion} | "
                f"embed_strength={self.embed_strength} | "
                f"ckpt={ckpt_short(self.eval_ckpt)}"
                + (
                    f" | epoch={self.embedding_epoch}"
                    if self.eval_ckpt == "99"
                    else ""
                )
                + f"\n         weights={weight_path}"
            )
    def print_network(self, model, name: str) -> None:
        """打印可训练参数量。"""
        num_params = sum(p.numel() for p in model.parameters())
        if not self.quiet:
            log_info(f"网络 [{name}] 参数量: {num_params:,}  device={self.device}")

    # ------------------------------------------------------------------
    # 可视化 / 指标辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _psnr_neg1_1(a: torch.Tensor, b: torch.Tensor) -> float:
        """PSNR for tensors in [-1, 1] (peak range = 2 → MAX=4 in MSE 公式)。"""
        mse = torch.mean((a.detach() - b.detach()) ** 2).item()
        if mse < 1e-10:
            return 99.0
        return float(10.0 * np.log10(4.0 / mse))

    @staticmethod
    def _psnr_neg1_1_mean(a: torch.Tensor, b: torch.Tensor) -> tuple[float, int]:
        """逐图 PSNR 再平均；返回 (mean_psnr, n_images)。"""
        if a.shape != b.shape or a.numel() == 0:
            return 0.0, 0
        mse = torch.mean((a.detach() - b.detach()) ** 2, dim=(1, 2, 3))
        # 10*log10(4/mse)；极小 MSE → 99
        psnr = torch.where(
            mse < 1e-10,
            torch.full_like(mse, 99.0),
            10.0 * torch.log10(4.0 / mse.clamp_min(1e-10)),
        )
        return float(psnr.mean().item()), int(a.shape[0])

    def _load_psnr_refs(
        self, nums, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        按文件名索引加载含水印宿主作为 PSNR 参考。

        Returns
        -------
        refs : FloatTensor [B,3,H,W] in [-1,1] 或 None（全部缺失）
        mask : BoolTensor [B] 哪些样本成功加载
        """
        if isinstance(nums, torch.Tensor):
            idx_list = [int(x) for x in nums.detach().cpu().view(-1).tolist()]
        else:
            idx_list = [int(x) for x in nums]

        ref_dir = self.psnr_ref_dir
        if not ref_dir.endswith(("/", "\\")):
            ref_dir = ref_dir + os.sep

        tensors = []
        mask = []
        for n in idx_list:
            path = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                cand = os.path.join(ref_dir, f"{n}{ext}")
                if os.path.isfile(cand):
                    path = cand
                    break
            if path is None:
                tensors.append(None)
                mask.append(False)
                continue
            img = cv2.imread(path, 1)
            if img is None:
                tensors.append(None)
                mask.append(False)
                continue
            img = cv2.resize(img, (self.image_size, self.image_size))
            arr = img.transpose((2, 0, 1)).astype(np.float32) / 255.0 * 2.0 - 1.0
            tensors.append(torch.from_numpy(arr))
            mask.append(True)

        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        if not any(mask):
            return None, mask_t
        # 缺失槽用零图占位，调用方用 mask 过滤
        zeros = torch.zeros(
            3, self.image_size, self.image_size, dtype=torch.float32
        )
        stacked = torch.stack(
            [t if t is not None else zeros for t in tensors], dim=0
        ).to(device)
        return stacked, mask_t

    @staticmethod
    def _bits_from_logits(decoded: torch.Tensor) -> torch.Tensor:
        """
        将解码器连续输出转为 {0,1} 比特。

        使用 ``>= 0.5`` 阈值（与原版 ``round().clip`` 在多数情况等价，
        但避免 numpy banker's round 把 0.5 收成 0 的偏差）。
        """
        return (decoded.detach() >= 0.5).float()

    @staticmethod
    def _bit_errors(pred_bits: torch.Tensor, target: torch.Tensor) -> float:
        """统计比特错误数（BER 分子）。变量名勿叫 correct。"""
        t = target.detach().float()
        return float((pred_bits - t).abs().sum().item())

    @staticmethod
    def _norm_map_to_uint8(chw: torch.Tensor) -> np.ndarray:
        """
        将任意范围的 CHW tensor 按 min-max 归一化到 uint8 HWC，避免过饱和白块/黑块。
        若近似常数图，输出中灰 128。
        """
        arr = chw.detach().float().cpu().numpy()
        if arr.ndim != 3:
            raise ValueError(f"期望 CHW，得到 shape={arr.shape}")
        # 跨通道统一归一化，保留相对结构
        amin = float(arr.min())
        amax = float(arr.max())
        if amax - amin < 1e-8:
            hwc = np.full((arr.shape[1], arr.shape[2], arr.shape[0]), 128.0, dtype=np.float32)
        else:
            norm = (arr - amin) / (amax - amin)
            hwc = np.transpose(norm, (1, 2, 0)) * 255.0
        return np.clip(hwc, 0, 255)

    @staticmethod
    def _residual_to_uint8(encoded_chw: torch.Tensor, host_chw: torch.Tensor, gain: float = 5.0) -> np.ndarray:
        """残差可视化：中灰为 0，增益放大后 min-max 归一化，正负差异都可见。"""
        res = (encoded_chw.detach() - host_chw.detach()).float().cpu().numpy() * gain
        amin = float(res.min())
        amax = float(res.max())
        if amax - amin < 1e-8:
            hwc = np.full((res.shape[1], res.shape[2], res.shape[0]), 128.0, dtype=np.float32)
        else:
            norm = (res - amin) / (amax - amin)
            hwc = np.transpose(norm, (1, 2, 0)) * 255.0
        return np.clip(hwc, 0, 255)

    @staticmethod
    def _to_uint8_bgr(tensor_chw: torch.Tensor) -> np.ndarray:
        """将 [-1,1] CHW tensor 转为 uint8 HWC（OpenCV BGR 友好）。"""
        arr = (tensor_chw.detach().to("cpu").numpy() + 1) / 2 * 255
        return np.transpose(arr, (1, 2, 0))

    def _save_train_panel(
        self,
        epoch: int,
        inputs,
        Encoded_image,
        Noised_image,
        mask,
        v_mask,
    ) -> None:
        """保存六联拼图：原图 | 编码 | 噪声 | 梯度mask | 视觉mask | 残差。"""
        t = np.random.randint(inputs.shape[0])
        I1 = self._to_uint8_bgr(inputs[t])
        I2 = self._to_uint8_bgr(Encoded_image[t])
        I_no = self._to_uint8_bgr(Noised_image[t])
        # mask / v_mask 用 min-max 归一化，避免 (v-1)*255 过饱和成白块
        I_mask = self._norm_map_to_uint8(mask[t])
        I_vmask = self._norm_map_to_uint8(v_mask[t])
        I_res = self._residual_to_uint8(Encoded_image[t], inputs[t], gain=5.0)
        w = I1.shape[1]
        panel = np.zeros((I1.shape[0], w * 6, I1.shape[2]), dtype=np.float32)
        panel[:, 0:w, :] = I1
        panel[:, w : w * 2, :] = I2
        panel[:, w * 2 : w * 3, :] = I_no
        panel[:, w * 3 : w * 4, :] = I_mask
        panel[:, w * 4 : w * 5, :] = I_vmask
        panel[:, w * 5 : w * 6, :] = I_res
        out_dir = os.path.join(self.result_dir, "Image", "images")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{epoch}.png")
        cv2.imwrite(out_path, np.clip(panel, 0, 255).astype(np.uint8))
        if self.verbose:
            log_step(f"可视化已保存: {out_path}")

    # ------------------------------------------------------------------
    # 嵌入
    # ------------------------------------------------------------------
    def test_embedding(self) -> None:
        """使用预训练 Encoder 嵌入水印，导出五联拼图。"""
        data_loader = self.data_loader
        criterion_MSE = nn.MSELoss()
        self.net_ED.eval()

        out_dir = os.path.join(
            self.result_dir,
            f"Image_test_{self.distortion}",
            f"images_embed_{self.embedding_epoch}",
        )
        os.makedirs(out_dir, exist_ok=True)

        total_batches = len(data_loader)
        with ProgressMonitor(
            total=total_batches,
            desc="Embedding",
            enable=self.show_progress,
            unit="batch",
            log_every=self.log_step,
        ) as mon:
            for data, m, num in data_loader:
                inputs, m = Variable(data), Variable(m.float())
                inputs, m = inputs.to(self.device), m.to(self.device)
                inputs.requires_grad = True
                num = num.to("cpu").numpy()

                Encoded_image, Noised_image, Decoded_message = self.net_ED(inputs, m)
                loss_de = criterion_MSE(Decoded_message, m)
                loss_de.backward()
                inputgrad = inputs.grad.data
                mask = torch.zeros(inputgrad.shape, device=self.device)
                for ii in range(inputgrad.shape[0]):
                    a = inputgrad[ii, :, :, :]
                    a = (1 - (a - a.min()) / (a.max() - a.min() + 1e-8)) + 1
                    mask[ii, :, :, :] = a

                for j in range(Encoded_image.shape[0]):
                    I1 = self._to_uint8_bgr(inputs[j])
                    I2 = self._to_uint8_bgr(Encoded_image[j])
                    I_no = self._to_uint8_bgr(Noised_image[j])
                    I_mask = self._norm_map_to_uint8(mask[j])
                    I_res = self._residual_to_uint8(Encoded_image[j], inputs[j], gain=5.0)
                    w = I1.shape[1]
                    panel = np.zeros((I1.shape[0], w * 5, I1.shape[2]), dtype=np.float32)
                    panel[:, 0:w, :] = I1
                    panel[:, w : w * 2, :] = I2
                    panel[:, w * 2 : w * 3, :] = I_no
                    panel[:, w * 3 : w * 4, :] = I_mask
                    panel[:, w * 4 : w * 5, :] = I_res
                    cv2.imwrite(
                        os.path.join(out_dir, f"{num[j]}.png"),
                        np.clip(panel, 0, 255).astype(np.uint8),
                    )

                mon.update(1, loss=float(loss_de.item()))

        log_ok(f"嵌入完成 → {out_dir}")

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def train_mask(self) -> None:
        """端到端训练 Encoder–Noise–Decoder + Discriminator。"""
        data_loader = self.data_loader
        data_loader_test = self.data_loader_test
        criterion = nn.BCEWithLogitsLoss()
        criterion_MSE = nn.MSELoss()
        start_epoch = self.embedding_epoch

        run_log = TrainRunLogger(self.log_dir, self)
        # 把实际算力设备写入 JSON，便于事后核对
        try:
            dev_info = {"device": str(self.device)}
            if self.device.type == "cuda" and torch.cuda.is_available():
                idx = self.device.index if self.device.index is not None else 0
                dev_info["gpu_name"] = torch.cuda.get_device_name(idx)
                dev_info["cuda_available"] = True
            else:
                dev_info["cuda_available"] = False
            run_log.doc["device"] = dev_info
            run_log._flush_json()
        except Exception:
            pass

        # 每次训练（全量 / Lite）独立权重目录，保留本 run 全部 epoch，互不覆盖
        run_ckpt_dir = os.path.join(self.model_save_dir, "runs", run_log.run_id)
        os.makedirs(run_ckpt_dir, exist_ok=True)
        distortion_ckpt_dir = os.path.join(self.model_save_dir, self.distortion)
        os.makedirs(distortion_ckpt_dir, exist_ok=True)
        run_info_path = os.path.join(run_ckpt_dir, "run_info.txt")
        with open(run_info_path, "w", encoding="utf-8") as rf:
            rf.write(f"run_id={run_log.run_id}\n")
            rf.write(f"distortion={self.distortion}\n")
            rf.write(f"lite={self.lite}\n")
            rf.write(f"num_epoch={self.num_epoch}\n")
            rf.write(f"embedding_epoch={self.embedding_epoch}\n")
            rf.write(f"embed_strength={self.embed_strength}\n")
            rf.write(f"model_save_step={self.model_save_step}\n")
            rf.write(f"init_from_ckpt={self.init_from_ckpt}\n")
            rf.write(f"init_from_epoch={self.init_from_epoch}\n")
        # 指针：最近一次训练权重目录（可覆盖）
        try:
            latest_ptr = os.path.join(self.model_save_dir, "LATEST_RUN.txt")
            with open(latest_ptr, "w", encoding="utf-8") as lf:
                lf.write(f"run_id={run_log.run_id}\n")
                lf.write(f"ckpt_dir={run_ckpt_dir}\n")
                lf.write(f"distortion={self.distortion}\n")
                lf.write(f"lite={self.lite}\n")
        except OSError:
            pass
        try:
            run_log.doc["ckpt_dir"] = run_ckpt_dir.replace("\\", "/")
            run_log._flush_json()
        except Exception:
            pass

        txtfile = run_log.txtfile
        if not self.quiet:
            log_info(f"训练日志(JSON) → {run_log.json_path}")
            log_info(f"训练日志(TXT)  → {run_log.txt_path}")
            log_info(f"run_id={run_log.run_id}（历史 run 永不覆盖）")
            log_ok(f"本轮权重目录 → {run_ckpt_dir}（每 epoch 存盘，不覆盖其它 run）")
            log_info(
                f"训练样本 {len(data_loader.dataset)} | "
                f"验证样本 {len(data_loader_test.dataset)} | "
                f"每 epoch 约 {len(data_loader)} batch"
            )
        run_log.set_dataset_info(
            train_samples=len(data_loader.dataset),
            val_samples=len(data_loader_test.dataset),
            batches_per_epoch=len(data_loader),
        )

        best_acc = 0.0
        best_epoch: int | None = None
        best_ckpt_path: str | None = None
        train_start = time.time()
        status = "completed"

        try:
            for epoch in range(start_epoch, self.num_epoch):
                # Warmup 结束：切回目标噪声层
                if (
                    self.distortion in ("ScreenShooting", "ScreenShootingMB")
                    and self.warmup_epochs > 0
                    and epoch == self.warmup_epochs
                    and start_epoch < self.warmup_epochs
                    and self.init_from_ckpt is None
                    and self.init_from_epoch is None
                ):
                    self._set_noiser(self.distortion)
                    msg = (
                        f"Warmup 结束 @ epoch {epoch}，启用 {self.distortion} + 完整感知损失 "
                        f"λ=({self.lambda1},{self.lambda2},{self.lambda3})"
                    )
                    log_ok(msg)
                    run_log.add_event("info", msg)

                # Identity 预热阶段：消息优先，避免 v_mask 放大项把 Acc 钉在 ~50%
                in_warmup = (
                    self.distortion in ("ScreenShooting", "ScreenShootingMB")
                    and self.warmup_epochs > 0
                    and epoch < self.warmup_epochs
                    and self.init_from_ckpt is None
                    and self.init_from_epoch is None
                )

                epoch_t0 = time.time()
                running_loss = 0.0
                n_logged = 0
                train_err = 0.0
                train_total = 0
                last_msg = 0.0
                last_den = 0.0
                last_loss = 0.0
                batch_loss_hist: list[float] = []
                batch_msg_hist: list[float] = []
                batch_den_hist: list[float] = []

                phase = "warmup" if in_warmup else "full"
                mon = ProgressMonitor(
                    total=len(data_loader),
                    desc=f"Epoch {epoch + 1}/{self.num_epoch}[{phase}]",
                    enable=self.show_progress,
                    unit="batch",
                    log_every=self.log_step,
                    file=txtfile,
                )

                for i, (data, m, v_mask) in enumerate(data_loader):
                    inputs, m, v_mask = Variable(data), Variable(m.float()), Variable(v_mask)
                    inputs = inputs.to(self.device)
                    m = m.to(self.device)
                    v_mask = v_mask.to(self.device)
                    if i == 0 and epoch == start_epoch and self.device.type == "cuda":
                        # 首 batch 核对张量确实在 NVIDIA 上
                        log_info(
                            f"首 batch 张量 device={inputs.device} | "
                            f"cuda_mem_alloc={torch.cuda.memory_allocated() / 1024**2:.1f} MB"
                        )

                    if in_warmup:
                        Encoded_image, Noised_image, Decoded_message = self.net(inputs, m)
                        loss_message = criterion_MSE(Decoded_message, m)
                        loss_image = criterion_MSE(Encoded_image, inputs)
                        loss = loss_message * max(self.lambda1, 10.0) + loss_image * 2.0
                        self.net_optimizer.zero_grad()
                        loss.backward()
                        self.net_optimizer.step()
                        mask = torch.ones_like(Encoded_image)
                        g_loss = torch.tensor(0.0, device=self.device)
                        d_loss = torch.tensor(0.0, device=self.device)
                        loss_denoise = loss_image
                    else:
                        inputs.requires_grad = True
                        Encoded_image, Noised_image, Decoded_message = self.net(inputs, m)
                        loss_de = criterion_MSE(Decoded_message, m)
                        inputgrad = torch.autograd.grad(
                            loss_de, inputs, create_graph=True
                        )[0]
                        mask = torch.zeros(inputgrad.shape, device=self.device)
                        for ii in range(inputgrad.shape[0]):
                            a = inputgrad[ii, :, :, :]
                            a = (1 - (a - a.min()) / (a.max() - a.min() + 1e-8)) + 1
                            mask[ii, :, :, :] = a.detach()

                        d_label_host = torch.full(
                            (inputs.shape[0], 1), 1, dtype=torch.float, device=self.device
                        )
                        d_label_decoded = torch.full(
                            (inputs.shape[0], 1), 0, dtype=torch.float, device=self.device
                        )
                        g_label_decoded = torch.full(
                            (inputs.shape[0], 1), 1, dtype=torch.float, device=self.device
                        )

                        self.optimizer_Discriminator.zero_grad()
                        d_image = self.net_Discriminator(inputs.detach())
                        d_loss_host = criterion(d_image, d_label_host)
                        d_loss_host.backward()
                        d_decoded = self.net_Discriminator(Encoded_image.detach())
                        d_loss = criterion(d_decoded, d_label_decoded)
                        d_loss.backward()
                        self.optimizer_Discriminator.step()

                        g_decoded = self.net_Discriminator(Encoded_image)
                        g_loss = criterion(g_decoded, g_label_decoded)
                        loss_message = criterion_MSE(Decoded_message, m)
                        loss_denoise = (
                            criterion_MSE(
                                Encoded_image * mask.float(), inputs * mask.float()
                            )
                            * 0.5
                            + criterion_MSE(
                                Encoded_image * v_mask.float(), inputs * v_mask.float()
                            )
                            * 2
                        )
                        loss = (
                            loss_message * self.lambda1
                            + loss_denoise * self.lambda2
                            + g_loss * self.lambda3
                        )
                        self.net_optimizer.zero_grad()
                        loss.backward()
                        self.net_optimizer.step()

                    with torch.no_grad():
                        pred_bits = self._bits_from_logits(Decoded_message)
                        train_err += self._bit_errors(pred_bits, m)
                        train_total += inputs.shape[0] * m.shape[1]

                    running_loss += loss.item()
                    n_logged += 1
                    last_loss = float(loss.item())
                    last_msg = float(loss_message.item())
                    last_den = float(loss_denoise.item())
                    batch_loss_hist.append(last_loss)
                    batch_msg_hist.append(last_msg)
                    batch_den_hist.append(last_den)
                    train_acc_so_far = (1.0 - train_err / max(train_total, 1)) * 100.0
                    mon.update(
                        1,
                        loss=loss.item(),
                        msg=loss_message.item(),
                        den=loss_denoise.item(),
                        acc=train_acc_so_far,
                    )

                    if (
                        not self.show_progress
                        and i % self.log_step == self.log_step - 1
                    ):
                        avg = running_loss / max(n_logged, 1)
                        line = (
                            f"[{epoch + 1}, {i + 1:5d}] loss: {avg:.3f} | "
                            f"msg:{loss_message.item():.3f} den:{loss_denoise.item():.3f} "
                            f"acc:{train_acc_so_far:.1f}%"
                        )
                        print(line, flush=True)
                        print(line, file=txtfile, flush=True)
                        running_loss = 0.0
                        n_logged = 0

                mon.close()
                train_bit_acc = (1.0 - train_err / max(train_total, 1)) * 100.0
                train_ber = train_err / max(train_total, 1)

                # ---- 本 epoch 曲线弹窗（msg / den / loss）----
                if not self.no_epoch_plot:
                    curve_dir = os.path.join(
                        self.result_dir, "Image", "epoch_curves", self.distortion
                    )
                    if not self.quiet:
                        log_info(
                            f"Epoch {epoch + 1} 曲线弹窗 "
                            f"(batches={len(batch_loss_hist)})，关闭窗口后继续…"
                        )
                    saved = show_epoch_curves(
                        epoch=epoch + 1,
                        num_epoch=self.num_epoch,
                        phase=phase,
                        loss_hist=batch_loss_hist,
                        msg_hist=batch_msg_hist,
                        den_hist=batch_den_hist,
                        save_dir=curve_dir,
                        block=self.epoch_plot_block,
                        enabled=True,
                    )
                    if saved is not None and self.verbose:
                        log_step(f"曲线已保存: {saved}")

                if self.save_viz:
                    self._save_train_panel(
                        epoch, inputs, Encoded_image, Noised_image, mask, v_mask
                    )

                if not self.quiet:
                    log_info(f"Epoch {epoch + 1} 验证中…")
                self.net.eval()
                bit_err = 0.0
                total = 0
                psnr_sum = 0.0
                psnr_n = 0
                val_iter = wrap_iterable(
                    data_loader_test,
                    total=len(data_loader_test),
                    desc=f"Val {epoch + 1}",
                    enable=self.show_progress,
                    leave=False,
                )
                with torch.no_grad():
                    for data, m, v_mask in val_iter:
                        inputs, m, v_mask = (
                            Variable(data),
                            Variable(m.float()),
                            Variable(v_mask),
                        )
                        inputs = inputs.to(self.device)
                        m = m.to(self.device)
                        Encoded_image, Noised_image, Decoded_message = self.net(inputs, m)
                        pred_bits = self._bits_from_logits(Decoded_message)
                        bit_err += self._bit_errors(pred_bits, m)
                        total += inputs.shape[0] * m.shape[1]
                        psnr_sum += self._psnr_neg1_1(Encoded_image, inputs)
                        psnr_n += 1

                bit_acc = (1.0 - bit_err / max(total, 1)) * 100.0
                bit_ber = bit_err / max(total, 1)
                psnr = psnr_sum / max(psnr_n, 1)
                epoch_sec = time.time() - epoch_t0
                remain_epochs = self.num_epoch - epoch - 1
                eta = format_eta(epoch_sec * remain_epochs)
                noise_tag = self._ed_module().distortion
                summary = (
                    f"Epoch {epoch + 1}/{self.num_epoch}  "
                    f"TrainAcc={train_bit_acc:.2f}%  ValAcc={bit_acc:.2f}%  "
                    f"BER={bit_ber:.4f} (train {train_ber:.4f})  "
                    f"PSNR={psnr:.2f}dB  noise={noise_tag}  "
                    f"time={format_eta(epoch_sec)}  ETA≈{eta}"
                )
                log_ok(summary)

                if psnr < 20 and not in_warmup:
                    w = (
                        f"PSNR={psnr:.1f}dB 偏低，嵌入可能过强；可增大 λ2 或减小 --embed_strength"
                    )
                    log_warn(w)
                    run_log.add_event("warn", w)
                if train_bit_acc < 55 and epoch + 1 >= 2 and in_warmup:
                    w = (
                        "Warmup 阶段 TrainAcc 仍接近随机。若持续低于 60%，"
                        "可略增大 --embed_strength 或 --lr。"
                    )
                    log_warn(w)
                    run_log.add_event("warn", w)

                os.makedirs(distortion_ckpt_dir, exist_ok=True)
                os.makedirs(run_ckpt_dir, exist_ok=True)

                # 文件名与历史一致：…_mask_{epoch}.pth；run 目录用零填充便于排序
                fname_epoch = f"{self.model_name}_mask_{epoch}.pth"
                fname_epoch_pad = f"{self.model_name}_mask_{epoch:04d}.pth"
                path_epoch_run = os.path.join(run_ckpt_dir, fname_epoch_pad)
                path_epoch_shared = os.path.join(distortion_ckpt_dir, fname_epoch)

                ckpt_saved = None
                # 本 run 目录：每个 epoch 都存（全量 / Lite 相同），互不覆盖其它 run
                state = self.net.state_dict()
                torch.save(state, path_epoch_run)
                ckpt_saved = path_epoch_run
                # 共享 distortion 目录：仍受 model_save_step 节流（便于旧续训路径）
                if epoch % self.model_save_step == (self.model_save_step - 1):
                    torch.save(state, path_epoch_shared)
                if self.verbose:
                    log_step(f"已保存: {path_epoch_run}")
                elif not self.quiet and (
                    epoch == start_epoch or epoch + 1 == self.num_epoch
                ):
                    log_info(f"已保存 epoch 权重: {path_epoch_run}")

                # 本 run 内 best + 全局命名 best（评估 CLI 仍读后者）
                path_best_run = os.path.join(
                    run_ckpt_dir,
                    f"{self.model_name}_mask_{self.distortion}_best.pth",
                )
                path_best_global = os.path.join(
                    self.model_save_dir,
                    f"{self.model_name}_mask_{self.distortion}_best.pth",
                )
                is_best = False
                if bit_acc / 100.0 >= best_acc:
                    best_acc = bit_acc / 100.0
                    best_epoch = int(epoch)
                    best_ckpt_path = path_best_run
                    is_best = True
                    torch.save(state, path_best_run)
                    torch.save(state, path_best_global)
                    for meta_base in (path_best_run, path_best_global):
                        meta_path = meta_base.replace(".pth", ".meta.txt")
                        with open(meta_path, "w", encoding="utf-8") as mf:
                            mf.write(f"embed_strength={self.embed_strength}\n")
                            mf.write(f"distortion={self.distortion}\n")
                            mf.write(f"val_acc={bit_acc:.6f}\n")
                            mf.write(f"epoch={epoch}\n")
                            mf.write(f"run_id={run_log.run_id}\n")
                            mf.write(f"ckpt_dir={run_ckpt_dir}\n")
                    log_ok(
                        f"新最佳 ValAcc={bit_acc:.3f}% (BER={bit_ber:.4f})\n"
                        f"         run  → {path_best_run}\n"
                        f"         全局 → {path_best_global}"
                    )

                run_log.log_epoch(
                    {
                        "epoch": int(epoch + 1),
                        "epoch_index": int(epoch),
                        "num_epoch": int(self.num_epoch),
                        "phase": phase,
                        "train_acc": float(train_bit_acc),
                        "train_ber": float(train_ber),
                        "val_acc": float(bit_acc),
                        "val_ber": float(bit_ber),
                        "psnr": float(psnr),
                        "noise": str(noise_tag),
                        "seconds": float(epoch_sec),
                        "time": format_eta(epoch_sec),
                        "eta": eta,
                        "is_best": bool(is_best),
                        "last_batch_loss": last_loss,
                        "last_batch_msg": last_msg,
                        "last_batch_den": last_den,
                        "batch_curve_points": len(batch_loss_hist),
                        "checkpoint": ckpt_saved,
                        "best_checkpoint": path_best_run if is_best else None,
                    }
                )

                self.net.train()

        except KeyboardInterrupt:
            status = "aborted"
            log_warn("训练被中断 (KeyboardInterrupt)，已保存当前 JSON 日志")
            run_log.add_event("warn", "KeyboardInterrupt — training aborted")
        except Exception as e:
            status = "failed"
            run_log.add_event("error", f"{type(e).__name__}: {e}")
            run_log.finalize(
                status=status,
                best_val_acc=best_acc,
                best_epoch=best_epoch,
                best_ckpt=best_ckpt_path,
                total_seconds=time.time() - train_start,
            )
            run_log.close()
            raise
        finally:
            if status != "failed":
                total_sec = time.time() - train_start
                run_log.finalize(
                    status=status,
                    best_val_acc=best_acc,
                    best_epoch=best_epoch,
                    best_ckpt=best_ckpt_path,
                    total_seconds=total_sec,
                )
                run_log.close()
                if status == "completed":
                    log_ok(
                        f"训练结束，总耗时 {format_eta(total_sec)}，"
                        f"最佳准确率 {best_acc * 100:.3f}% | run_id={run_log.run_id}\n"
                        f"         全部 epoch 权重 → {run_ckpt_dir}"
                    )
                else:
                    log_warn(
                        f"训练未正常结束 status={status} | run_id={run_log.run_id} | "
                        f"日志仍已保存: {run_log.json_path}\n"
                        f"         已写入权重目录 → {run_ckpt_dir}"
                    )

    # ------------------------------------------------------------------
    # 跳过训练：加载权重直接评估
    # ------------------------------------------------------------------
    def eval_mask(self) -> None:
        """
        加载权重，在验证集上跑 Encoder → Noiser → Decoder，报告 Acc / BER / PSNR，
        导出可视化，并生成「本次评估结果 | 训练最后一轮 result」对比图。
        """
        data_loader_test = self.data_loader_test
        self.net.eval()

        bit_err = 0.0
        total = 0
        psnr_sum = 0.0
        psnr_n = 0
        viz_saved = 0
        max_viz = 12
        last_panel = None  # 最后一轮（最后一个 batch）可视化

        out_dir = os.path.join(
            self.result_dir,
            f"Image_eval_{self.distortion}",
            self._eval_run_tag(),
        )
        if self.save_viz:
            os.makedirs(out_dir, exist_ok=True)

        log_info(
            f"评估中：val batches={len(data_loader_test)} | "
            f"noise={self._ed_module().distortion} | "
            f"embed_strength={self.embed_strength}"
            + (f" | viz→{out_dir}" if self.save_viz else "")
        )

        val_iter = wrap_iterable(
            data_loader_test,
            total=len(data_loader_test),
            desc="Eval",
            enable=self.show_progress,
            leave=True,
        )
        n_batches = len(data_loader_test)
        with torch.no_grad():
            for bi, (data, m, v_mask) in enumerate(val_iter):
                inputs, m, v_mask = (
                    Variable(data),
                    Variable(m.float()),
                    Variable(v_mask),
                )
                inputs = inputs.to(self.device)
                m = m.to(self.device)
                v_mask = v_mask.to(self.device)
                Encoded_image, Noised_image, Decoded_message = self.net(inputs, m)
                pred_bits = self._bits_from_logits(Decoded_message)
                bit_err += self._bit_errors(pred_bits, m)
                total += inputs.shape[0] * m.shape[1]
                psnr_sum += self._psnr_neg1_1(Encoded_image, inputs)
                psnr_n += 1

                is_last_batch = bi == n_batches - 1
                if self.save_viz and (viz_saved < max_viz or is_last_batch):
                    j_limit = inputs.shape[0] if is_last_batch else min(
                        inputs.shape[0], max_viz - viz_saved
                    )
                    for j in range(j_limit):
                        I1 = self._to_uint8_bgr(inputs[j])
                        I2 = self._to_uint8_bgr(Encoded_image[j])
                        I_no = self._to_uint8_bgr(Noised_image[j])
                        I_abs = self._norm_map_to_uint8(
                            (Encoded_image[j] - inputs[j])
                            .abs()
                            .mean(dim=0, keepdim=True)
                            .expand_as(inputs[j])
                        )
                        I_vmask = self._norm_map_to_uint8(v_mask[j])
                        I_res = self._residual_to_uint8(
                            Encoded_image[j], inputs[j], gain=5.0
                        )
                        w = I1.shape[1]
                        panel = np.zeros(
                            (I1.shape[0], w * 6, I1.shape[2]), dtype=np.float32
                        )
                        panel[:, 0:w, :] = I1
                        panel[:, w : w * 2, :] = I2
                        panel[:, w * 2 : w * 3, :] = I_no
                        panel[:, w * 3 : w * 4, :] = I_abs
                        panel[:, w * 4 : w * 5, :] = I_vmask
                        panel[:, w * 5 : w * 6, :] = I_res
                        panel_u8 = np.clip(panel, 0, 255).astype(np.uint8)

                        sample_err = float(
                            (pred_bits[j] - m[j]).abs().mean().item()
                        )
                        sample_acc = (1.0 - sample_err) * 100.0

                        if viz_saved < max_viz:
                            out_path = os.path.join(
                                out_dir, f"{bi:04d}_{j}_acc{sample_acc:.0f}.png"
                            )
                            cv2.imwrite(out_path, panel_u8)
                            viz_saved += 1

                        if is_last_batch and j == 0:
                            last_panel = panel_u8
                            last_path = os.path.join(out_dir, "last_eval_result.png")
                            # 顶部标注
                            labeled = panel_u8.copy()
                            cv2.putText(
                                labeled,
                                f"LAST EVAL batch  Acc={sample_acc:.1f}%",
                                (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 180, 0),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.imwrite(last_path, labeled)
                            last_panel = labeled

        bit_acc = (1.0 - bit_err / max(total, 1)) * 100.0
        bit_ber = bit_err / max(total, 1)
        psnr = psnr_sum / max(psnr_n, 1)
        weight_name = (
            os.path.basename(self.eval_weight_path)
            if self.eval_weight_path
            else ckpt_label(self.eval_ckpt)
        )
        summary = (
            f"[eval_mask] ckpt={weight_name}  "
            f"ValAcc={bit_acc:.2f}%  BER={bit_ber:.4f}  "
            f"PSNR={psnr:.2f}dB  noise={self._ed_module().distortion}  "
            f"embed_strength={self.embed_strength}  bits={int(total)}"
        )
        log_ok(summary)

        # CLI 结果面板
        panel_txt = (
            "\n"
            "  ┌─────────────── 评估结果 ───────────────┐\n"
            f"  │  权重           {weight_name:<22s}│\n"
            f"  │  ValAcc         {bit_acc:>6.2f}%               │\n"
            f"  │  BER            {bit_ber:>8.4f}             │\n"
            f"  │  PSNR           {psnr:>6.2f} dB             │\n"
            f"  │  噪声层         {self._ed_module().distortion:<22s}│\n"
            f"  │  embed_strength {self.embed_strength:<22g}│\n"
            f"  │  比特总数       {int(total):<22d}│\n"
        )
        if self.save_viz:
            panel_txt += (
                f"  │  可视化         {viz_saved} 张 → Image_eval_*   │\n"
            )
        panel_txt += "  └─────────────────────────────────────────┘\n"
        print(panel_txt, flush=True)

        summary_path = None
        if self.save_viz:
            # 指标看板 + 本次最后一轮评估 + 训练最后一轮 result
            board = self._draw_metrics_board(bit_acc, bit_ber, psnr)
            train_last_path = self._find_last_train_result_image()
            train_last_img = None
            if train_last_path and os.path.isfile(train_last_path):
                train_last_img = cv2.imread(train_last_path, 1)
                log_info(f"找到训练最后一轮 result: {train_last_path}")
            else:
                log_warn(
                    "未找到 results/Image/images/<epoch>.png；"
                    "对比图仅含本次评估结果（可先训练一轮生成）"
                )

            # 两行布局：上行指标；下行 评估最后一轮 | 训练最后一轮
            row0 = board
            row1_imgs = []
            if last_panel is not None:
                row1_imgs.append(last_panel)
            if train_last_img is not None:
                tagged = train_last_img.copy()
                cv2.putText(
                    tagged,
                    "LAST TRAIN result",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 140, 255),
                    1,
                    cv2.LINE_AA,
                )
                row1_imgs.append(tagged)
            if row1_imgs:
                row1 = self._hconcat_resize(row1_imgs, row_h=256)
                # 对齐宽度
                w = max(row0.shape[1], row1.shape[1])
                def pad_w(img, ww):
                    if img.shape[1] >= ww:
                        return img
                    pad = np.full(
                        (img.shape[0], ww - img.shape[1], 3), 230, dtype=np.uint8
                    )
                    return np.concatenate([img, pad], axis=1)
                row0 = pad_w(row0, w)
                row1 = pad_w(row1, w)
                summary_img = np.concatenate([row0, row1], axis=0)
            else:
                summary_img = row0

            summary_path = os.path.join(out_dir, "summary_eval_and_last_train.png")
            cv2.imwrite(summary_path, summary_img)
            log_ok(f"对比总览已保存: {summary_path}")
            self._try_open_image(summary_path)
            if last_panel is not None:
                self._try_open_image(os.path.join(out_dir, "last_eval_result.png"))

        out_txt = os.path.join(
            self.log_dir, f"eval_mask_{self.distortion}_{self._eval_run_tag()}.txt"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
            if self.save_viz:
                f.write(f"viz_dir={out_dir}\n")
                f.write(f"viz_count={viz_saved}\n")
                f.write(f"last_eval={os.path.join(out_dir, 'last_eval_result.png')}\n")
                if summary_path:
                    f.write(f"summary={summary_path}\n")
                train_last_path = self._find_last_train_result_image()
                if train_last_path:
                    f.write(f"last_train_result={train_last_path}\n")
        log_ok(f"评估结果已写入: {out_txt}")
        if self.save_viz:
            log_ok(
                f"可视化：样本拼图 + last_eval_result + summary_eval_and_last_train → {out_dir}"
            )

    # ------------------------------------------------------------------
    # 测准
    # ------------------------------------------------------------------
    def test_accuracy(self) -> None:
        """
        对拍屏矫正后的图像评估比特准确率，并计算相对含水印宿主的 PSNR。

        - Acc / BER：Decoder(Recover) vs w.mat
        - PSNR：Recover vs ``psnr_ref_dir`` 中同索引含水印图（默认 images_watermarked）
        """
        data_loader_test = self.data_loader_test
        bit_err = 0.0
        total = 0
        psnr_sum = 0.0
        psnr_n = 0
        psnr_missing = 0
        warned_missing = False

        ref_dir = self.psnr_ref_dir
        if not os.path.isdir(ref_dir):
            log_warn(
                f"PSNR 参考目录不存在: {ref_dir}；将只报告 Acc/BER。"
                "可用 --psnr_ref_dir 指定含水印宿主目录（如 Datasets/images_watermarked/）"
            )

        with ProgressMonitor(
            total=len(data_loader_test),
            desc="Accuracy",
            enable=self.show_progress,
            unit="batch",
            log_every=max(1, self.log_step // 2),
        ) as mon:
            for data, m, num in data_loader_test:
                inputs, m = Variable(data), Variable(m.float())
                inputs, m = inputs.to(self.device), m.to(self.device)
                self.net_D.eval()
                with torch.no_grad():
                    Decoded_message = self.net_D(inputs)
                    pred_bits = self._bits_from_logits(Decoded_message)
                    if self.verbose:
                        print(pred_bits.detach().cpu().numpy())
                    bit_err += self._bit_errors(pred_bits, m)
                    total += inputs.shape[0] * m.shape[1]

                    if os.path.isdir(ref_dir):
                        refs, mask = self._load_psnr_refs(num, self.device)
                        if refs is not None and mask is not None and bool(mask.any()):
                            mean_p, n_ok = self._psnr_neg1_1_mean(
                                inputs[mask], refs[mask]
                            )
                            psnr_sum += mean_p * n_ok
                            psnr_n += n_ok
                            n_miss = int((~mask).sum().item())
                            psnr_missing += n_miss
                            if n_miss and not warned_missing:
                                nums_cpu = (
                                    num.detach().cpu().view(-1)
                                    if isinstance(num, torch.Tensor)
                                    else torch.as_tensor(num).view(-1)
                                )
                                miss_ex = [
                                    int(nums_cpu[i])
                                    for i in range(nums_cpu.numel())
                                    if not bool(mask.cpu()[i])
                                ][:5]
                                log_warn(
                                    f"部分索引在 PSNR 参考目录中缺失（例: {miss_ex}），"
                                    "已跳过这些样本的 PSNR"
                                )
                                warned_missing = True
                        else:
                            psnr_missing += int(inputs.shape[0])
                            if not warned_missing:
                                log_warn(
                                    f"未能加载任何 PSNR 参考图（目录 {ref_dir}）"
                                )
                                warned_missing = True

                acc_so_far = (1.0 - bit_err / max(total, 1)) * 100.0
                extra = {"acc": acc_so_far, "ber": bit_err / max(total, 1)}
                if psnr_n > 0:
                    extra["psnr"] = psnr_sum / psnr_n
                mon.update(1, **extra)

        final_acc = (1.0 - bit_err / max(total, 1)) * 100.0
        final_ber = bit_err / max(total, 1)
        final_psnr = psnr_sum / psnr_n if psnr_n > 0 else float("nan")

        weight_name = (
            os.path.basename(self.eval_weight_path)
            if self.eval_weight_path
            else ckpt_label(self.eval_ckpt)
        )
        if psnr_n > 0:
            log_ok(
                f"Accuracy ({self.distortion}, {weight_name}): {final_acc:.3f}%  "
                f"BER={final_ber:.4f}  PSNR={final_psnr:.2f}dB  "
                f"(vs {ref_dir}, n={psnr_n}"
                + (f", miss={psnr_missing}" if psnr_missing else "")
                + ")"
            )
        else:
            log_ok(
                f"Accuracy ({self.distortion}, {weight_name}): {final_acc:.3f}%  BER={final_ber:.4f}  "
                f"PSNR=N/A（无参考图）"
            )

        panel = (
            "\n"
            "  ┌─────────────── 拍屏测准结果 ─────────────┐\n"
            f"  │  权重           {weight_name:<22s}│\n"
            f"  │  Acc            {final_acc:>6.2f}%               │\n"
            f"  │  BER            {final_ber:>8.4f}             │\n"
        )
        if psnr_n > 0:
            panel += f"  │  PSNR           {final_psnr:>6.2f} dB             │\n"
            panel += f"  │  PSNR样本       {psnr_n:<22d}│\n"
        else:
            panel += "  │  PSNR           N/A                    │\n"
        panel += (
            f"  │  噪声层         {self.distortion:<22s}│\n"
            f"  │  比特总数       {int(total):<22d}│\n"
            "  └─────────────────────────────────────────┘\n"
            f"  PSNR ref: {os.path.normpath(ref_dir)}\n"
        )
        print(panel, flush=True)
        if self.save_viz and psnr_n > 0:
            out_dir = os.path.join(self.result_dir, "Image_accuracy")
            os.makedirs(out_dir, exist_ok=True)
            board = self._draw_metrics_board(final_acc, final_ber, final_psnr)
            cv2.putText(
                board,
                "Screen-shooting Accuracy",
                (36, board.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )
            out_path = os.path.join(out_dir, "accuracy_metrics.png")
            cv2.imwrite(out_path, board)
            summary_path = os.path.join(out_dir, "last_accuracy_result.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(
                    f"Acc={final_acc:.4f}%\nBER={final_ber:.6f}\n"
                    f"PSNR={final_psnr:.4f}dB\nn={psnr_n}\n"
                    f"ref={ref_dir}\ndistortion={self.distortion}\n"
                    f"ckpt={weight_name}\n"
                )
            log_info(f"测准指标看板 → {out_path}")
