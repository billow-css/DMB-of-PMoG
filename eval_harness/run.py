#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness.run
================
统一评估入口。

既可作为独立 CLI 使用：:

    python -m eval_harness.run --mode sim --models pimog,dmb_pmog,st_rep

也可被 ``main.py`` 在进程内调用（``main.run_eval_harness`` → 本模块 ``run_harness``）。
核心逻辑集中在 ``run_harness(...)``，CLI 的 ``main()`` 只是薄薄一层参数转译。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from eval_harness import registry
from eval_harness import adapters  # noqa: F401  触发注册
from eval_harness import datasets, metrics, report


def _to_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _fmt(v, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (v != v):  # NaN
        return "—"
    return f"{v:.{nd}f}"


def _files(d: str, num_samples):
    files = datasets.list_images(d)
    return files[:num_samples] if num_samples else files


def _parse_kv(s: str) -> dict:
    """``"name=path,name2=path2"`` → ``{name: path}``。"""
    out = {}
    if not s:
        return out
    for kv in s.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# sim：encode → native_noise → decode
# ---------------------------------------------------------------------------
def run_sim(models, host_dir: str, seed: int, native_noise: bool, num_samples):
    files = _files(host_dir, num_samples)
    rows = []
    for model in models:
        bit_err, total_bits = 0.0, 0
        psnr_sum, psnr_n = 0.0, 0
        ssim_sum, ssim_n = 0.0, 0
        for idx, fname in enumerate(files):
            x = datasets.load_canonical(
                os.path.join(host_dir, fname), model.spec.input_size
            ).to(model.device)
            m = torch.from_numpy(
                datasets.generate_message(idx, model.spec.message_bits, seed)
            ).unsqueeze(0).to(model.device)
            with torch.no_grad():
                enc = model.encode(x, m)
                n = model.native_noise(enc) if native_noise else None
                dec_in = n if n is not None else enc
                logits = model.decode(dec_in)
            pred = metrics.bits_from_logits(logits)
            bit_err += metrics.bit_errors(pred, m)
            total_bits += m.numel()
            p = metrics.psnr_neg1_1_mean(enc, x)
            if p == p:  # 非 NaN
                psnr_sum += p
                psnr_n += 1
            ssim_sum += metrics.ssim_rgb(enc[0], x[0])
            ssim_n += 1

        acc = (1.0 - bit_err / max(total_bits, 1)) * 100
        ber = bit_err / max(total_bits, 1)
        row = {
            "model": model.spec.name,
            "label": model.spec.label,
            "message_bits": model.spec.message_bits,
            "input_size": model.spec.input_size,
            "acc": f"{acc:.2f}%",
            "ber": f"{ber:.4f}",
            "psnr": _fmt(psnr_sum / max(psnr_n, 1)),
            "ssim": _fmt(ssim_sum / max(ssim_n, 1), 4),
            "n_images": len(files),
            "noise": "native" if native_noise else "none",
        }
        rows.append(row)
        print(f"[{model.spec.name}] Acc={acc:.2f}%  BER={ber:.4f}  "
              f"PSNR={row['psnr']}  SSIM={row['ssim']}  (n={len(files)})")
    return rows


# ---------------------------------------------------------------------------
# test：只 decode（拍屏模拟）
# ---------------------------------------------------------------------------
def run_test(models, test_dir: str, msg_matrix: str, num_samples):
    files = _files(test_dir, num_samples)
    W = datasets.load_message_matrix(msg_matrix)
    rows = []
    for model in models:
        if model.spec.message_bits > W.shape[1]:
            raise ValueError(
                f"{model.spec.name} 需要 {model.spec.message_bits} bit，"
                f"但消息矩阵 {msg_matrix} 只有 {W.shape[1]} 列"
            )
        bit_err, total_bits = 0.0, 0
        for fname in files:
            idx = datasets.filename_index(fname)
            x = datasets.load_canonical(os.path.join(test_dir, fname)).to(model.device)
            with torch.no_grad():
                logits = model.decode(x)
            pred = metrics.bits_from_logits(logits)
            m = torch.from_numpy(
                W[idx % W.shape[0], : model.spec.message_bits]
            ).unsqueeze(0).float().to(model.device)
            bit_err += metrics.bit_errors(pred, m)
            total_bits += m.numel()

        acc = (1.0 - bit_err / max(total_bits, 1)) * 100
        ber = bit_err / max(total_bits, 1)
        row = {
            "model": model.spec.name,
            "label": model.spec.label,
            "message_bits": model.spec.message_bits,
            "input_size": model.spec.input_size,
            "acc": f"{acc:.2f}%",
            "ber": f"{ber:.4f}",
            "psnr": "—",
            "ssim": "—",
            "n_images": len(files),
            "noise": "real_screen_shot",
        }
        rows.append(row)
        print(f"[{model.spec.name}] Acc={acc:.2f}%  BER={ber:.4f}  (n={len(files)})")
    return rows


# ---------------------------------------------------------------------------
# embed：产出含水印图 + 真相消息（供人工拍屏后跑 test）
# ---------------------------------------------------------------------------
def run_embed(models, host_dir: str, seed: int, out_dir: str, num_samples):
    import cv2

    files = _files(host_dir, num_samples)
    os.makedirs(out_dir, exist_ok=True)
    for model in models:
        mdir = os.path.join(out_dir, model.spec.name)
        os.makedirs(mdir, exist_ok=True)
        truths = {}
        for idx, fname in enumerate(files):
            x = datasets.load_canonical(
                os.path.join(host_dir, fname), model.spec.input_size
            ).to(model.device)
            mbits = datasets.generate_message(idx, model.spec.message_bits, seed)
            m = torch.from_numpy(mbits).unsqueeze(0).to(model.device)
            with torch.no_grad():
                enc = model.encode(x, m)
            arr = ((enc[0].detach().cpu().numpy() + 1) / 2 * 255).clip(0, 255)
            arr = arr.transpose(1, 2, 0).astype(np.uint8)  # BGR（与 load_canonical 一致）
            out_path = os.path.join(mdir, f"{idx}.png")
            cv2.imwrite(out_path, arr)
            truths[str(idx)] = mbits.tolist()
        with open(os.path.join(mdir, "truth.json"), "w", encoding="utf-8") as f:
            json.dump(truths, f)
        print(f"[{model.spec.name}] 已嵌入 {len(files)} 张 → {mdir}")


# ---------------------------------------------------------------------------
# 核心：统一入口（供 CLI 与 main.py 共同调用）
# ---------------------------------------------------------------------------
def run_harness(
    mode: str = "sim",
    models: str = "pimog,dmb_pmog,st_rep",
    device: str = "auto",
    host_dir: str = None,
    test_dir: str = None,
    msg_matrix: str = None,
    out_dir: str = None,
    seed: int = 0,
    num_samples: int = None,
    native_noise: bool = True,
    weights: str = None,
    ckpts: str = None,
    list_only: bool = False,
) -> int:
    dev = _to_device(device)

    if list_only:
        print("可用模型:", ", ".join(registry.available()))
        return 0

    names = [n.strip() for n in models.split(",") if n.strip()]
    weights_map = _parse_kv(weights)
    ckpts_map = _parse_kv(ckpts)

    built = []
    for n in names:
        kw = {}
        if n in weights_map:
            kw["weight_path"] = weights_map[n]
        if n in ckpts_map:
            kw["ckpt_tag"] = ckpts_map[n]
        m = registry.build(n, **kw)
        m.to(dev)
        built.append(m)

    if mode == "sim":
        rows = run_sim(
            built, host_dir or str(REPO_ROOT / "Datasets" / "images"),
            seed, native_noise, num_samples,
        )
    elif mode == "test":
        rows = run_test(
            built,
            test_dir or str(REPO_ROOT / "Datasets" / "Recover" / "capture"),
            msg_matrix or str(REPO_ROOT / "results" / "WatermarkMatrix" / "w.mat"),
            num_samples,
        )
    elif mode == "embed":
        run_embed(
            built, host_dir or str(REPO_ROOT / "Datasets" / "images"),
            seed, out_dir or str(REPO_ROOT / "results" / "eval_harness"), num_samples,
        )
        return 0
    else:
        print(f"未知 mode: {mode}")
        return 2

    md = report.write_report(
        rows, mode, out_dir or str(REPO_ROOT / "results" / "eval_harness"),
        extra={"device": str(dev), "seed": seed, "models": ",".join(names)},
    )
    print(f"\n报告: {md}")
    return 0


# ---------------------------------------------------------------------------
# 独立 CLI（python -m eval_harness.run）
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval_harness", description="统一水印模型评估")
    p.add_argument("--mode", choices=["sim", "test", "embed"], default="sim")
    p.add_argument("--models", default="pimog,dmb_pmog,st_rep",
                   help="逗号分隔；可用: pimog,dmb_pmog,st_rep,stegastamp,ropass,sim2real")
    p.add_argument("--host_dir", default=None)
    p.add_argument("--test_dir", default=None)
    p.add_argument("--msg_matrix", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--no_native_noise", action="store_true",
                   help="sim 模式下禁用各模型自带噪声层")
    p.add_argument("--weights", default=None, help="name=path,name2=path2 覆盖默认权重")
    p.add_argument("--ckpt", default=None,
                   help="name=tag,name2=tag 选自研模型权重标签（99/ss_best/best/mb_best）")
    p.add_argument("--device", default="auto")
    p.add_argument("--list", action="store_true", help="列出可用模型后退出")
    return p


def main(argv=None) -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    return run_harness(
        mode=args.mode,
        models=args.models,
        device=args.device,
        host_dir=args.host_dir,
        test_dir=args.test_dir,
        msg_matrix=args.msg_matrix,
        out_dir=args.out_dir,
        seed=args.seed,
        num_samples=args.num_samples,
        native_noise=not args.no_native_noise,
        weights=args.weights,
        ckpts=args.ckpt,
        list_only=args.list,
    )


if __name__ == "__main__":
    raise SystemExit(main())
