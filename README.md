# EE6008：AAPL 下一交易日收益预测

本项目用苹果股票和 Nasdaq 市场历史信息，预测 AAPL 下一交易日的收盘收益率，并辅助评价涨跌方向。小组实验约定、成员待办和需要同步的信息见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

## 1. 当前已经完成什么

- 从原始 CSV 构造共同数据集：每天 8 个特征、最近 60 个交易日作为输入。
- 实现 XGBoost、PatchTST 单目标适配模型，以及零收益、训练期平均收益、延续前一日收益三种基线。
- 完成首轮训练和验证；按日期保存预测、指标、配置、训练记录、预处理参数和模型权重。
- 核心代码使用 `.py`，Notebook 调用相同模块探索数据和展示结果。
- 统一启动命令是 `python main.py`。当前入口完成训练和验证，尚未实现最终测试入口、LSTM 或多模型结果自动汇总。

训练期为 2010—2021 年，验证期为 2022—2023 年；2024—2025 年最终测试期及 2026 年额外保留期尚未评价。验证结果用于选择模型，不作为最终测试成绩。

## 2. 配置环境

以下流程适用于 Windows / Linux x86-64，使用 conda 管理 Python、pip 安装项目包。先安装 [Miniforge](https://github.com/conda-forge/miniforge)，再打开能够使用 `conda` 的终端。已有可运行的 EE6008 环境可以直接跳到第 3 节。

### 2.1 打开项目并创建环境

把终端工作目录切换到项目根目录，也就是能看到 `main.py` 的目录。如果终端当前在项目的上一级，可运行：

```bash
cd stock_prediction
```

从空环境开始：

```bash
conda create -n EE6008 python=3.12 pip -y
conda activate EE6008
```

如果同名环境已经存在，使用现有环境，或者为新环境换一个名称，并在后续激活、内核选择时使用该名称。项目不依赖特定环境名称。

### 2.2 安装 PyTorch：下面两种方式选一种

**有兼容 NVIDIA GPU 和驱动，使用 GPU 训练：**

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
```

**没有 NVIDIA GPU，或准备使用 CPU：**

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

本项目已经在 Windows、RTX 5070 Ti Laptop GPU、PyTorch 2.14.0+cu132 上运行过。CPU 与 CUDA 13.2 安装包可在 [CPU 官方索引](https://download.pytorch.org/whl/cpu/torch/) 和 [CUDA 13.2 官方索引](https://download.pytorch.org/whl/cu132/torch/) 查询。其他硬件或操作系统请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择对应构建，并记录差异；不能只根据本机 CUDA Toolkit 版本判断 GPU 能否运行。

项目只使用 `torch`，不需要 `torchvision` 或 `torchaudio`。

### 2.3 安装其余依赖

```bash
python -m pip install -r requirement.txt
python -m pip check
```

保留现有文件名 `requirement.txt`。其中固定了 PyTorch、NumPy、pandas、scikit-learn、matplotlib、XGBoost 和 Notebook 内核 ipykernel 的版本；先前安装的匹配 CPU / CUDA 构建满足 PyTorch 的版本要求。它是直接依赖清单，不是全部间接依赖的完整锁文件。

### 2.4 检查运行环境

```bash
python -c "import torch, numpy, pandas, sklearn, matplotlib, xgboost; print('torch:', torch.__version__); print('xgboost:', xgboost.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CPU 安装下 `CUDA: False` 是正常结果。准备使用 GPU 时，应显示 `CUDA: True`，并进一步确认能实际进行 GPU 运算：

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.randn(32,32,device='cuda'); print((x @ x).mean().item()); torch.cuda.synchronize()"
```

默认 XGBoost 使用 CPU，PatchTST 的 `device=auto` 在 CUDA 可用时使用 GPU，否则使用 CPU。

## 3. 准备数据

向小组获取同一版本的数据，并保留下面的相对位置：

```text
stock_prediction/
└── data/
    └── data1/
        └── AAPL_Apple_stock_data.csv
```

当前训练只读取这份原始 CSV。它包含 AAPL 和 Nasdaq 的行情字段，代码会重新计算特征；无需把 `with_features.csv` 或 `standardized.csv` 作为训练输入。TSLA 尚未接入当前实验配置。

检查文件是否存在，并打印文件指纹：

```bash
python -c "from pathlib import Path; import hashlib; p=Path('data/data1/AAPL_Apple_stock_data.csv'); print(p.exists()); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

当前参考数据的 SHA-256 为：

```text
5628d6886f2b3da80fd3ace8884ef90876c1a664a7eaa82ca90e5d952dff44cc
```

不同成员的指纹不一致时，先确认数据版本再比较结果。数据来源、Nasdaq 具体代码和价格复权口径仍待组员补充。

## 4. 完成一次实验：检查 → 训练 → 查看 → 提交

以下命令都在项目根目录执行，并确保已经 `conda activate EE6008`。

### 4.1 运行项目检查

```bash
python -m unittest discover -s tests -v
```

这些检查验证标签时间对齐、训练期标准化、无效数据处理、指标和模型保存恢复等。这里的“代码测试”不等于在 2024—2025 年测试集上评价模型。

### 4.2 启动一次完整训练与验证

```bash
python main.py
```

这一条命令会依次完成：

1. 读取 `configs/aapl.json` 和原始 CSV，在构造特征前排除验证期之后的数据。
2. 构造 60 日窗口，按目标日划分训练和验证样本，计算训练期预处理统计量。
3. 生成简单基线，训练 XGBoost 和 PatchTST，并根据验证表现选择最佳轮次。
4. 计算验证集收益误差和方向指标。
5. 将本次实验全部结果写入一个新的 `artifacts/validation_时间戳/` 目录。

默认配置下应得到 2,960 条训练样本、501 条验证样本。由于需要积累历史窗口，第一个训练目标日是 2010-04-01；验证目标日为 2022-01-03 至 2023-12-29。

终端会打印数据检查摘要、训练进度、指标表，最后显示 `Results:` 和结果目录。`manifest.json` 中 `status` 为 `complete` 才表示整次实验完成。

### 4.3 常用运行方式

```bash
python main.py --help
python main.py --models baselines
python main.py --models xgboost
python main.py --models patchtst
python main.py --models xgboost patchtst --run-name aapl_compare_01
python main.py --config configs/aapl.json
```

- `--models` 选择训练模型；每次都会同时输出简单基线。
- `--run-name` 指定新的结果文件夹名称，只使用英文字母、数字、下划线或连字符。同名目录已存在时会报错，换一个新名称即可；不指定则自动生成时间戳名称。
- `--config` 指定配置文件。在根目录运行时使用项目相对路径即可。
- 调整参数时，建议复制 `configs/aapl.json` 为自己的配置文件，通过 `--config` 指定；一次只改变准备研究的参数，并记录目的。
- `python scripts/run_experiment.py` 仍兼容；日常统一使用根目录 `main.py`，两者调用同一套逻辑。

现在完成的是一轮“训练 + 验证”实验。配置中的测试期日期是后续约定，当前程序不会因此自动评价最终测试集。

### 4.4 查看结果：先看这三项

| 文件 | 先关注什么 |
|---|---|
| `metrics.csv` | 比较各模型 RMSE、MAE（越低越好）及方向准确率 |
| `validation_comparison.png` | 直观看误差和方向表现 |
| `predictions.csv` | 每个目标日的真实收益和各模型预测，用于分析及小组合并 |

`mae`、`rmse` 使用小数收益单位，例如误差 `0.01` 对应 1 个百分点。图中已乘以 100。方向由收益是否大于 0 得到，平盘归入“不涨”。回归误差小与方向准确率高是两个不同目标，应分别报告。

完整结果包还包含：

| 文件 | 作用 |
|---|---|
| `metrics.json` | 完整指标和混淆矩阵；矩阵行是真实类别、列是预测类别，顺序为 `[不涨, 上涨]` |
| `direction_reference.json` | 用训练期多数方向作预测时的验证准确率 |
| `data_audit.json` | 样本数量、日期边界及无效窗口记录 |
| `preprocessing.json` | 特征顺序、训练均值和标准差、目标缩放参数 |
| `config.json` | 本次运行的配置快照，应以它解释对应实验 |
| `manifest.json` | 是否完成、数据和代码指纹、环境版本、随机种子及评价分区 |
| `training.json` | 训练耗时、设备、最佳轮次和学习曲线 |
| `xgboost.ubj` / `patchtst.pt` | 本次训练模型的最佳权重，需配合预处理参数使用；只生成所选模型文件 |

`r2` 相对验证集真实均值计算，`r2_vs_zero = 1-SSE/sum(y²)` 相对零收益预测计算，不能混称。小组比较时保留简单基线作为参照。

### 4.5 使用 Notebook 展示

用支持 Notebook 的编辑器打开 `notebooks/01_explore_and_results.ipynb`，选择刚配置的 Python 环境内核，再从上到下运行。

- Notebook 读取结果，不会自动训练模型。
- 默认选择最近完成的验证实验，包括仅运行基线的实验。要展示某一轮，在结果读取单元格设置 `RUN_NAME = "aapl_compare_01"`，名称与结果目录一致。
- 数据探索部分读取当前 `configs/aapl.json`，结果部分读取该次实验的配置快照；两者不一致时会显示 `Config matches current exploration: False`，应区分它们。
- 若编辑器找不到内核，可在激活环境后运行 `python -m ipykernel install --user --name EE6008 --display-name "Python (EE6008)"`，再选择该内核；自定义环境名时同步替换。
- 历史验证中曾遇到 ZeroMQ 内核启动错误，编辑器内核仍需各成员本机检查；可先通过 CSV 和 PNG 阅读完整实验结果。

### 4.6 向小组提交

提交模型代码、所用配置和一个 `status=complete` 的完整实验结果目录，并简要说明实验目的、改动和结论。不要只提交截图或一个准确率数字。具体输出约定见 [项目规划](PROJECT_PLAN.md)。

`data/`、`artifacts/`、模型权重、课程文档和 `private/` 已被 Git 忽略。共享代码仓库不会自动共享数据和实验输出，需要通过小组约定的位置单独提供数据与选定的结果包。

## 5. 文件结构

```text
stock_prediction/
├── main.py                   # 日常使用的实验入口
├── README.md                 # 当前进展和操作手册
├── PROJECT_PLAN.md           # 共同约定、成员任务、待同步事项
├── requirement.txt           # Python 直接依赖版本
├── configs/aapl.json         # 日期、窗口、种子和模型参数
├── data/                     # 小组共享的原始数据，本地放置
├── src/
│   ├── data.py               # 共同的数据处理
│   ├── baselines.py          # 简单预测基线
│   ├── evaluate.py           # 共同的指标和绘图
│   └── models/               # XGBoost、PatchTST；后续接入其他模型
├── scripts/run_experiment.py # 实验执行和结果保存逻辑
├── notebooks/                # 探索与结果展示
├── tests/                    # 代码正确性检查
├── artifacts/                # 各次实验输出
└── private/                  # 个人笔记，不提交 Git
```

根目录只保留两份共享 Markdown。新增个人分析放入 `private/`；共同运行信息和约定分别维护在 README 与 PROJECT_PLAN 中。

## 6. 当前实现的边界

PatchTST 参考 [官方实现](https://github.com/yuqinie98/PatchTST) 的时间分块及通道共享编码思路，以 PyTorch 独立实现；当前使用 LayerNorm，不使用 RevIN、残差注意力或预训练，通过联合回归头输出单个 AAPL 次日收益。结果应标注为本项目适配版本。

目前只完成固定时间划分、单随机种子的首轮比较；尚未完成 LSTM、滚动验证、多种子、最终测试或回测。数据复权与来源未核实，当前收益按 `close` 计算，不宣称为含分红总收益。跨硬件或软件版本的结果可能存在差异，应保留环境记录。
