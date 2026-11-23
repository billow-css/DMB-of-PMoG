# 屏摄噪声层

`Noise_Layer.py` 里同时有抖动、透视、光照和摩尔纹。抖动函数写在文件前半，拍屏函数写在后半，`ScreenShootingMB` 按「抖动 → 透视 → 光照/摩尔纹 → 高斯」调用，不再从别的文件导入抖动再套一层。

```bash
python main.py --dataset train_mask --mode train_mask --distortion ScreenShootingMB
```

`model.py` 是编码器、解码器和判别器。`solver.py` 负责训练、嵌入和测准。噪声层在 `Noise_Layer.py`：`ScreenShootingMB` 依次做抖动、透视、光照/摩尔纹、高斯。

几何和参数见 [`docs/运动模糊噪声层.md`](docs/运动模糊噪声层.md)。

依赖：`torch`、`numpy`、`Pillow`、`kornia`。

## License

[MIT](LICENSE)。
