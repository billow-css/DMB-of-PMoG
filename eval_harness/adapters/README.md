# eval_harness 适配器

一个 adapter = 一个「把某模型塞进统一接口」的文件。统一接口（`eval_harness/base.py`）：

```python
class WatermarkModel(ABC):
    spec: ModelSpec                       # name / message_bits / input_size / label …
    def encode(self, x, m) -> Tensor      # [-1,1] 图 + 0/1 消息 → 含水印图（可选）
    def decode(self, x) -> Tensor         # [-1,1] 图 → 消息 logits [B, bits]
    def native_noise(self, x) -> Tensor|None  # 电脑模拟用的自带噪声层
```

**harness 规范**：图像统一 `float32 [B,3,H,W]` 范围 `[-1,1]`，通道序 **BGR（cv2）**；消息
`float32 [B,bits]` 取值 0/1；`decode` 输出「未取整的 logits」，由框架统一 `round` 后比对。
adapter 内部自行缩放分辨率 / 转换数值域（RGB 原生模型在 adapter 里做 BGR↔RGB 转换）。

## 已有模型

| 注册名 | 模型 | 消息 | 输入 | 状态 |
|:---|:---|:---:|:---:|:---|
| `pimog` | PIMoG（`ScreenShooting`） | 30 | 128×128 | ✅ |
| `dmb_pmog` | DMB-PMoG（`ScreenShootingMB`） | 30 | 128×128 | ✅ |
| `st_rep` | 风格迁移 + RepConv（第三方） | 30 | 128×128 | ✅ |
| `stegastamp` | StegaStamp | 100 | 400×400 | TODO |
| `ropass` | RoPaSS | L | 128×128 | TODO |
| `sim2real` | SIM2Real | 32 | 128×128 | TODO |

## 新增一个第三方模型要做的事（5 步）

1. **建文件**：复制 `stegastamp.py` 骨架为 `adapters/<name>.py`，改类名与 `@registry.register("<name>")`。
2. **在 `adapters/__init__.py` 里 import 该模块**，触发注册。
3. **填 `ModelSpec`**：`message_bits`、`input_size=(H,W)`、`label`；若模型只能「仅解码/预嵌入图」，
   设 `supports_encode=False`；若消息为模型全局固定，设 `fixed_message=True`。
4. **实现 `encode/decode/native_noise`**：
   - 输入图像用 `resize_canonical(x, self.spec.input_size)` 缩放后喂给网络；
   - 权重加载用 `load_state_dict_strip(net, path, device)`（自动剥 `module.` 前缀 + `state_dict` 包装）；
   - 若第三方代码与仓库根同名文件冲突（`model.py/Noise_Layer.py/…`），照抄 `st_rep.py` 的
     `_load_tp_module` 别名动态加载 + `sys.path` 临时注入，加载完移除，防串包。
5. **校准消息真相格式**：`sim` 模式由框架用 `seed + index` 生成消息（可复现）；`test` 模式读
   `results/WatermarkMatrix/w.mat`（30 列）。若你的模型消息长度 ≠30 或有自己的消息格式，
   在 `run.py` 的 `run_test` 里为该模型单独提供消息矩阵（或扩展 `--msg_matrix`）。

## 两种评估模式

- `sim`（电脑模拟）：`encode → native_noise → decode`，报 BER/bit-Acc + 嵌入图相对宿主的 PSNR/SSIM。
- `test`（拍屏模拟）：对矫正后的拍屏图只 `decode`，与真相消息比对，报 BER/bit-Acc（无需噪声层）。

## 命名空间隔离（重要）

第三方 `model.py/Noise_Layer.py/data_loader.py/solver.py` 与仓库根同名。`st_rep.py` 用
`importlib.util.spec_from_file_location("st_rep_model", …)` 等别名加载，并在 `exec` 前后对
`Noise_Layer` / `module` / `modules` / `distoration_model` 做 save/remove/restore，保证
**同一进程里自研与第三方互不串包**。新增第三方模型若存在同名文件，务必沿用该做法。
