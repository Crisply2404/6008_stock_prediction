# EE6008：单股票下一交易日收益预测

本项目用个股和 Nasdaq 市场历史信息，预测下一交易日的收盘收益率，并辅助评价涨跌方向。目前已有 AAPL 和 TSLA 实验。小组实验约定、成员待办和需要同步的信息见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

GitHub 分支开发、提交、PR 审查和冲突处理步骤见 [小组协作指南](cooperation.md)。

## 1. 当前已经完成什么

- 从原始 CSV 构造共同数据集，支持每天 8 或 6 个特征、最近 60 个交易日作为输入；本轮组员调参统一使用 8 特征。
- 实现 XGBoost、LightGBM、PatchTST、LSTM、GRU 单目标适配模型，以及零收益、训练期平均收益、延续前一日收益三种基线。
- 完成首轮训练和验证；按日期保存预测、指标、配置、训练记录、预处理参数和模型权重。
- 核心代码使用 `.py`，Notebook 调用相同模块探索数据和展示结果。
- 统一启动命令是 `python main.py`。当前入口完成训练和验证，可将选定模型写入同一结果表，尚未实现最终测试入口或跨实验结果自动汇总。
- LSTM 和 GRU 已按组员提供的模型和参数接入，使用共同数据处理、评价和结果输出。

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

保留现有文件名 `requirement.txt`。其中固定了 PyTorch、NumPy、pandas、scikit-learn、matplotlib、XGBoost、LightGBM 和 Notebook 内核 ipykernel 的版本；先前安装的匹配 CPU / CUDA 构建满足 PyTorch 的版本要求。它是直接依赖清单，不是全部间接依赖的完整锁文件。

### 2.4 检查运行环境

```bash
python -c "import torch, numpy, pandas, sklearn, matplotlib, xgboost, lightgbm; print('torch:', torch.__version__); print('xgboost:', xgboost.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CPU 安装下 `CUDA: False` 是正常结果。准备使用 GPU 时，应显示 `CUDA: True`，并进一步确认能实际进行 GPU 运算：

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.randn(32,32,device='cuda'); print((x @ x).mean().item()); torch.cuda.synchronize()"
```

默认 XGBoost、LightGBM 使用 CPU，PatchTST、LSTM 和 GRU 的 `device=auto` 在 CUDA 可用时使用 GPU，否则使用 CPU。仅运行树模型或基线时，启动入口不会导入 PyTorch。

## 3. 准备数据

数据随仓库共享，克隆或更新仓库后保留下面的相对位置：

```text
stock_prediction/
└── data/
    └── AAPL_Apple_stock_data.csv
```

默认配置读取 AAPL 原始 CSV；`configs/tsla.json` 读取 `data/TSLA_Tesla_stock_data.csv`。两者都包含个股和 Nasdaq 行情，代码重新计算特征；无需把 `with_features.csv` 或 `standardized.csv` 作为训练输入。

检查文件是否存在，并打印文件指纹：

```bash
python -c "from pathlib import Path; import hashlib; p=Path('data/AAPL_Apple_stock_data.csv'); print(p.exists()); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

当前参考数据的 SHA-256 为：

```text
5628d6886f2b3da80fd3ace8884ef90876c1a664a7eaa82ca90e5d952dff44cc
```

不同成员的指纹不一致时，先确认数据版本再比较结果。数据来源、Nasdaq 具体代码和价格复权口径仍待组员补充。

## 4. 完成本地实验：修改 JSON → 运行模型 → 查看结果

以下命令均在项目根目录（包含 `main.py`）执行，并先激活环境：

```bash
conda activate EE6008
```

### 4.1 运行项目检查

首次配置环境或修改代码后，可运行：

```bash
python -m unittest discover -s tests -v
```

检查集中在 `tests/test_models.py`，验证数据对齐、标准化、指标及模型接口等。代码检查不等于模型预测效果评价，也不会使用最终测试集。

### 4.2 修改 JSON 配置

**先选择股票配置，再修改自己负责模型的参数：** AAPL 使用 `configs/aapl.json`，TSLA 使用 `configs/tsla.json`。每个文件包含公共数据设置和五个模型的参数段。

本轮小组调参统一基于 **8 特征**, 开始前请确保的字段为：

```json
"nasdaq_features": "all"
```


**公共配置项：**

| 字段 | 含义与修改方法 |
|---|---|
| `ticker` / `data_path` | 股票名称 / 原始 CSV 的项目相对路径；两者应对应。换股票优先用 `--config` 选择对应文件，不能只改股票名称 |
| `nasdaq_features` | `"all"` 为 8 特征，`"return_volume"` 为 6 特征，具体见下表 |
| `start` / `train_end` / `validation_end` | 数据起始日、训练结束日、验证结束日；按预测目标日期划分。本轮调参保持原设置 |
| `test_end` / `holdout_end` | 最终测试期及额外保留期的约定日期；当前入口不会因此执行测试评价 |
| `window` | 历史输入交易日数，目前为 60；本轮调参保持一致 |
| `seed` | 随机种子，目前为 42；本轮比较保持一致，多种子实验另行约定 |

**切换 8 / 6 特征只需修改 `nasdaq_features`：**

| 配置值 | 个股特征 | Nasdaq 特征 | 总数 |
|---|---|---|---|
| `"all"` | `return`, `vol_change`, `intraday_return`, `range` | 同样四项，加 `nasdaq_` 前缀 | 8 |
| `"return_volume"` | 同上 | `nasdaq_return`, `nasdaq_vol_change` | 6 |

`nasdaq_vol_change = nasdaq_volume[t] / nasdaq_volume[t-1] - 1`，是成交量变化率。两种配置使用同一原始 CSV，无需手动删列；标准化仍由代码根据训练数据完成。

**模型参数项：只编辑自己负责的模型段，保留其他段落。**

| JSON 段落 | 常用可调参数及含义 |
|---|---|
| `xgboost` | `max_depth`：树深；`min_child_weight`：子节点最小权重约束；`n_estimators`：最大树数量；`learning_rate`：学习率；`subsample` / `colsample_bytree`：样本 / 特征抽样比例；`reg_lambda`：L2 正则化强度 |
| `lightgbm` | `num_leaves` / `max_depth`：树复杂度；`min_child_samples`：叶节点最少样本数；`n_estimators`、`learning_rate`、抽样比例、`reg_alpha` / `reg_lambda`：L1 / L2 正则化强度 |
| `patchtst` | `patch_len` / `stride`：时间片段长度 / 步长；`d_model`：表示维度；`n_heads`：注意力头数；`n_layers`：编码层数；`d_ff`：前馈网络宽度 |
| `lstm` / `gru` | `hidden_size`：隐藏状态维度；`num_layers`：循环网络层数 |
| 三种神经网络共用 | `learning_rate`：学习率；`dropout`：训练时随机置零部分激活的比例；`weight_decay`：权重衰减；`batch_size`：每批样本数；`epochs`：最大训练轮数；`patience`：验证表现连续不改善多少轮后停止 |

树模型的 `early_stopping_rounds` 同样控制验证早停。`n_jobs` 控制树模型 CPU 并行线程数；神经网络的 `device` 可选 `"auto"`、`"cpu"`、`"cuda"`，使用 CUDA 需要环境和硬件支持。当前树模型配置使用 CPU。

参数需满足实现约束：PatchTST 要求 `d_model` 能被 `n_heads` 整除，`0 < stride <= patch_len <= window`，且 `(window - patch_len)` 能被 `stride` 整除；GRU 若设 `num_layers=1`，需同时设 `dropout=0`。

例如负责 XGBoost 的成员，可以把 `xgboost.max_depth` 从 3 改为 2，保存后运行一次；下一次修改参数后使用新的结果目录名。**本轮保持数据、8 特征、日期、60 日窗口及种子一致，只调整模型参数**。先保留一轮原参数结果作为基准，再按计划逐次修改并记录，以验证集 RMSE 为主、MAE 和方向表现为辅选择配置；最终测试集继续封存。

### 4.3 运行实验与命令行参数

JSON 保存后即可运行，不必把 JSON 中的设置重复写到命令行。以 AAPL 的单模型调参为例，按自己负责的模型选择一条命令：

```bash
python main.py --config configs/aapl.json --models xgboost --run-name aapl_xgboost_tune_01
python main.py --config configs/aapl.json --models lightgbm --run-name aapl_lightgbm_tune_01
python main.py --config configs/aapl.json --models patchtst --run-name aapl_patchtst_tune_01
python main.py --config configs/aapl.json --models lstm --run-name aapl_lstm_tune_01
python main.py --config configs/aapl.json --models gru --run-name aapl_gru_tune_01
```

TSLA 只需选择对应配置，例如：

```bash
python main.py --config configs/tsla.json --models xgboost --run-name tsla_xgboost_tune_01
```

| 参数 | 作用 |
|---|---|
| `--config` | 指定 JSON 配置路径；省略时读取 `configs/aapl.json`。也可读取自己复制、编辑的完整配置文件 |
| `--models` | 选择模型，可填写一个或多个：`xgboost lightgbm patchtst lstm gru`；省略时运行全部五个。`baselines` 表示只运行简单基线；任何模型实验都会附带简单基线 |
| `--run-name` | 指定 `artifacts/` 下的新结果目录名；仅允许英文字母、数字、下划线和连字符。不能与已有目录重名；省略时自动生成时间戳名称 |
| `--nasdaq-features` | 可选，临时覆盖 JSON 的特征设置，取值 `all` 或 `return_volume`；覆盖后的值会写入本次结果快照，不修改原配置文件 |
| `--help` | 查看入口支持的参数 |

优先级为 **命令行 `--nasdaq-features` > JSON 中的 `nasdaq_features` > 未设置时默认 `all`**。例如 JSON 已设为 `"all"` 就不需要额外参数；需要临时强制使用 8 特征时可以运行：

```bash
python main.py --config configs/aapl.json --models xgboost --nasdaq-features all --run-name aapl_xgboost_tune_02
```

模型超参数、窗口和种子均在 JSON 中修改；当前入口没有 `--learning-rate`、`--window` 或 `--seed` 等选项。`python main.py` 会按当前 AAPL 配置运行全部五个模型；成员只调自己的模型时，请明确填写 `--models`。

每次运行执行训练与验证，将实际配置、预测、指标和权重保存到 `artifacts/<run-name>/`。终端应正常退出，并且结果中的 `manifest.json` 应显示 `status=complete`。后续修改源 JSON 不会改变已保存的实验；查看历史结果应以该目录的 `config.json` 为准。

### 4.4 查看结果：先看这三项

更新代码后，Notebook 建议先重启内核，再运行全部单元格。出现 `Dataset has no attribute 'features'` 时，通常是内核缓存了旧版数据模块，并不意味着旧实验缺少结果。展示代码会在重建数据集前重新加载本项目的 `src.data`；旧配置未含 `nasdaq_features` 时仍按八特征读取。若提示模块来自另一个项目目录，需重启内核并从当前项目重新运行。

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
| `xgboost.ubj` / `lightgbm.txt` / `patchtst.pt` / `lstm.pt` / `gru.pt` | 本次训练模型的最佳权重，需配合预处理参数使用；只生成所选模型文件 |

`r2` 相对验证集真实均值计算，`r2_vs_zero = 1-SSE/sum(y²)` 相对零收益预测计算，不能混称。小组比较时保留简单基线作为参照。

### 4.5 使用 Notebook 展示

单次实验可用 `notebooks/01_AAPL_explore_and_results.ipynb` 或 `notebooks/01_TSLA_explore_and_results.ipynb` 展示。两者使用相同的通用读取规则，文件名仅用于整理，不限制所展示的股票。单模型调参请设置 `REQUIRE_ALL_MODELS = False`。

在开头**必须填写**实验目录名，例如：

```python
RUN_NAME = "aapl_xgboost_tune_01"
REQUIRE_ALL_MODELS = False
```

展示另一轮实验时，只需将 `RUN_NAME` 改为 `artifacts/` 下实际存在的目录名称（例如 `"tsla_xgboost_tune_01"`），再从头运行。不要填写完整路径、`None` 或空字符串。目录不存在或文件不完整时直接报错，不会自动选择其他实验。

- 股票名称由 `artifacts/<RUN_NAME>/config.json` 的 `ticker` 获取；数据路径、日期范围、窗口、随机种子和模型参数也来自这份快照。
- Notebook 不读取 `configs/` 下当前的训练配置，不按目录名称猜测股票，也不扫描最新实验。加载后会显示实际股票、实验目录和数据路径，图表标题及图例随之更新。
- `REQUIRE_ALL_MODELS = True` 要求所选实验包含五种模型；查看单模型或部分模型实验时设为 `False`。
- 读取 `predictions.csv`、`metrics.json`、`training.json`、`manifest.json`、`preprocessing.json` 和 `direction_reference.json`，核对原始数据指纹、日期、真实收益及指标。只展示已完成的验证实验，不重新训练。
- 依次展示数据和模型参数、误差与方向指标、对比图、预测曲线、累计平方误差、学习曲线、耗时及混淆矩阵；结论来自所选实验的实际数值。
- `RECENT_DAYS` 只控制局部图展示的验证日数，不影响历史输入窗口或全期指标。`SHOW_TREE_IMPORTANCE = True` 可额外读取已有树模型权重。
- 选择 EE6008 内核后从头运行。修改 `RUN_NAME` 文字不会自动更新旧图表；如一直转圈，可先单独执行 `print("kernel ready")` 检查内核是否响应。
- 原始 CSV 必须仍位于快照指定的位置，且与训练时数据指纹一致。新增股票只要遵循现有公共数据格式和实验输出约定，就能复用任一本 Notebook；改变特征定义或预测任务仍需同步公共代码。

现有参考配置下，AAPL 为 2,960 条训练样本、501 条验证样本；TSLA 为 2,838 条训练样本、501 条验证样本。实际数量应以所选实验记录为准。

### 4.6 向小组提交

提交模型代码、所用配置和一个 `status=complete` 的完整实验结果目录，并简要说明实验目的、改动和结论。不要只提交截图或一个准确率数字。具体输出约定见 [项目规划](PROJECT_PLAN.md)。

`data/` 中的六份 CSV 随仓库共享；每次训练读取所选配置指定的原始 CSV，其余文件供核对和后续实验使用。`artifacts/`、模型权重、课程文档和 `private/` 已被 Git 忽略，选定的实验结果包需要通过小组约定的位置单独共享。

## 5. 文件结构

```text
stock_prediction/
├── main.py                   # 日常使用的实验入口
├── README.md                 # 当前进展和操作手册
├── PROJECT_PLAN.md           # 共同约定、成员任务、待同步事项
├── cooperation.md            # GitHub 分支与 PR 协作步骤
├── requirement.txt           # Python 直接依赖版本
├── configs/                 # aapl.json、tsla.json：股票数据及五模型参数
├── data/                     # 六份共享 CSV，直接放在该目录
├── src/
│   ├── data.py               # 共同的数据处理
│   ├── baselines.py          # 简单预测基线
│   ├── evaluate.py           # 共同的指标和绘图
│   └── models/               # XGBoost、LightGBM、PatchTST、LSTM、GRU
├── scripts/run_experiment.py # 实验执行和结果保存逻辑
├── notebooks/               # 01_AAPL / 01_TSLA：均按指定实验目录通用展示
├── tests/test_models.py      # 统一的数据、指标及模型检查
├── artifacts/                # 各次实验输出
└── private/                  # 个人笔记，不提交 Git
```

共享文档分工：README 说明运行方法，PROJECT_PLAN 记录共同约定，cooperation 说明 GitHub 协作步骤。新增个人分析放入 `private/`。

## 6. 当前实现的边界

PatchTST 参考 [官方实现](https://github.com/yuqinie98/PatchTST) 的时间分块及通道共享编码思路，以 PyTorch 独立实现；当前使用 LayerNorm，不使用 RevIN、残差注意力或预训练，通过联合回归头输出单个目标股票次日收益。结果应标注为本项目适配版本。

LSTM 使用组员提供的单向两层结构，以最后一个时间步的输出经过回归头预测次日收益；GRU 使用单向两层结构，以最后一层的最终隐藏状态经过线性回归头输出收益。共享 Notebook 可展示所有已接入模型的预测曲线。`lstm/`、`gru/` 是原始交付目录，保留用于追溯并已忽略 Git 提交；实际运行使用 `src/models/` 和 `configs/`。

目前只完成固定时间划分、单随机种子的首轮比较；尚未完成滚动验证、多种子、最终测试或回测。数据复权与来源未核实，当前收益按 `close` 计算，不宣称为含分红总收益。跨硬件或软件版本的结果可能存在差异，应保留环境记录。
