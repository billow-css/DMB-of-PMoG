#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_harness
============
统一水印模型评估框架。

把「异构水印模型」（自研 PIMoG / DMB-PMoG，第三方 ST-Rep / StegaStamp / RoPaSS / SIM2Real …）
统一成一套公共接口（encode / decode / native_noise），再用一套公共指标
（BER / bit-Acc / PSNR / SSIM）跑「电脑模拟（sim）」与「拍屏模拟（test）」两类评估。

用法见 ``adapters/README.md``，入口：

    python -m eval_harness.run --mode sim --models pimog,dmb_pmog,st_rep
"""

__version__ = "0.1.0"
