# DMB of PMoG

面向屏幕翻拍场景的鲁棒数字水印研究代码。DMB（Dynamic Motion Blur）在 PIMoG 拍屏噪声的基础上加入可微运动模糊，模拟曝光期间的平移与旋转。

当前模型采用 Encoder–Noise–Decoder 结构，训练时加入判别器；默认宿主图为 **128×128**，水印消息为 **30 bit**。

```text
宿主图 + 30-bit 消息 → Encoder → 含水印图 → 噪声层 → Decoder → 恢复消息
```

## 噪声层

- `ScreenShootingMB`：DMB 主噪声层，依次施加运动模糊、透视、光照/摩尔纹、高斯噪声。
- `ScreenShooting`：拍屏基线，包含透视、光照/摩尔纹、高斯噪声。
- `Identity`：恒等映射，用于无失真对照。

运动模糊和拍屏失真均在 [`Noise_Layer.py`](Noise_Layer.py) 中实现。当前 `ScreenShootingMB` 在一次前向中为整个 batch 使用同一组运动位姿，并根据模糊强度降低摩尔纹权重。完整定义见[运动模糊噪声层](docs/运动模糊噪声层.md)。

## 环境

使用 Python 3.10 或更高版本，在仓库根目录执行。先安装与计算设备匹配的 PyTorch，再安装其余依赖：

```bash
python -m pip install torch
python -m pip install numpy scipy opencv-python Pillow kornia matplotlib tqdm
```

`matplotlib` 用于训练曲线，`tqdm` 用于进度显示；其余为训练、数据加载和噪声层所需依赖。代码自动选择可用的 CUDA 设备，否则使用 CPU。当前仓库未提供锁定版本的依赖清单。

## 快速开始

训练前需自行准备 COCOMask 数据。默认目录为：

```text
Datasets/COCOMask/train/train_class/
Datasets/COCOMask/val/val_class/
```

每张样本为横向拼接的 `[宿主图 | mask]`，默认高 128、宽 256；数据不会由程序自动下载或生成。

```bash
# 从头训练 DMB，使用默认数据目录
python main.py --dataset train_mask --mode train_mask --distortion ScreenShootingMB

# 使用随仓库提供的 DMB 权重评估验证集
python main.py --dataset eval_mask --mode eval_mask --distortion ScreenShootingMB --weight_path models/Encoder_Decoder_Model_mask_ScreenShootingMB_best.pth --image_dir Datasets/COCOMask/val/val_class/ --image_val_dir Datasets/COCOMask/val/val_class/

# 查看当前入口实际支持的参数（需先安装依赖）
python main.py --help
```

`--dataset` 决定数据格式及模型构建方式，`--mode` 决定执行函数；常规使用时应设为相同值。当前入口会同时加载 `--image_dir` 和 `--image_val_dir`，两者都须存在。

`python main.py` 默认直接进入训练。数据格式、基线评估、嵌入与拍屏测准见[运行指南](docs/运行指南.md)。

## 随仓库提供的权重

- `models/Encoder_Decoder_Model_mask_ScreenShootingMB_best.pth`：DMB 权重。
- `models/Encoder_Decoder_Model_mask_ScreenShooting_best.pth`：ScreenShooting 基线权重。
- `models/Encoder_Decoder_Model_mask_99.pth`：保留的 `mask_99` 权重。

评估、嵌入和测准时使用 `--weight_path` 明确指定文件。该参数不会为训练模式加载初始化权重。权重本身不足以确定实验条件；比较结果时还应记录数据、噪声配置及评估设置。

## 代码结构

- [`main.py`](main.py)：命令行入口与模式分发。
- [`model.py`](model.py)：编码器、解码器、判别器。
- [`Noise_Layer.py`](Noise_Layer.py)：运动模糊、拍屏噪声及辅助函数。
- [`solver.py`](solver.py)：训练、验证、嵌入、测准和输出管理。
- [`data_loader.py`](data_loader.py)：COCOMask 与带文件名索引的测试数据加载。
- [`ckpt_utils.py`](ckpt_utils.py)：权重名称与路径解析。
- [`cli/`](cli/) 与 [`cli_utils.py`](cli_utils.py)：日志、进度及保留的交互辅助代码。
- [`ui/`](ui/)：训练曲线、训练日志及辅助界面。
- [`docs/`](docs/README.md)：运行指南、算法定义及历史数据集说明。

数据集、日志、结果和大部分训练权重由 `.gitignore` 排除。当前仓库已移除 `app/`、`AS_VALs/`、`tools/` 和 `reference/`，文档中的运行流程使用现有入口。

## 文档

- [文档索引](docs/README.md)
- [运行指南：数据、命令、输出与指标](docs/运行指南.md)
- [运动模糊噪声层：公式、参数与实现边界](docs/运动模糊噪声层.md)
- [AS_VALs 数据集：历史规格与待核验项](docs/AS_VALs数据集.md)

## 许可证

仓库的 [`LICENSE`](LICENSE) 为 **GNU General Public License v3.0**。许可证以该文件为准。
