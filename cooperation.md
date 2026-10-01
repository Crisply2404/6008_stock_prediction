# GitHub 小组协作指南

本指南用于六人小组共享股票预测代码、接入不同模型和合并实验成果。项目运行方法见 [README](README.md)，共同实验规则见 [PROJECT_PLAN](PROJECT_PLAN.md)。

基本流程：**更新主分支 → 创建工作分支 → 开发与验证 → 提交并推送 → 创建 PR → 审查并合并。**

以下命令由成员手动执行。示例假设主分支为 `main`；仓库地址中的 `YOUR_USERNAME`、`YOUR_REPOSITORY` 需要替换。除首次克隆外，命令均在项目根目录执行。

## 1. 仓库维护者先做什么

在 GitHub 仓库的 Settings → Collaborators 中邀请组员。组员接受邀请后，用自己的 GitHub 账号向自己的工作分支推送代码。

共同约定：日常修改通过 PR 合并到 `main`。PR 是 Pull Request，即请求把工作分支的修改合并到主分支。提交 PR 后，由另一位成员检查，初期可由公共流程维护者负责整合。

## 2. 组员首次获取项目

在准备存放项目的父目录打开终端：

```bash
git clone https://github.com/screenpandar/6008_stock_prediction
cd 6008_stock_prediction
git config user.name "你的姓名或GitHub用户名"
git config user.email "你的GitHub提交邮箱"
```

`user.name` 和 `user.email` 是提交署名，不是登录信息。邮箱可以使用 GitHub 提供的 noreply 地址。推送时按 Git 的认证提示登录自己的账号，不共享账号或访问令牌。

按 README 从零创建 conda 环境并安装依赖，然后检查公共流程：

```bash
conda activate EE6008
python -m unittest discover -s tests -v
python main.py --models baselines
```

若使用自定义环境名，替换 `EE6008`。数据随仓库共享，当前入口使用 `data/AAPL_Apple_stock_data.csv`；基线运行应得到 2,960 条训练样本和 501 条验证样本。

## 3. 开始一项新任务

先查看工作区：

```bash
git status
```

如果存在未提交修改，先在所属工作分支整理并提交，避免把上一项任务的改动带进新分支。工作区干净后执行：

```bash
git switch main
git pull --ff-only origin main
git switch -c model/lstm-xiaowang
```

这三步分别是切换主分支、同步最新代码、创建并进入工作分支。示例中的分支名应换成自己本次任务的名称。

分支命名示例：

| 工作 | 分支示例 |
|---|---|
| 添加 LSTM | `model/lstm-xiaowang` |
| 添加 GRU | `model/gru-xiaoli` |
| 修改实验说明 | `docs/experiment-guide` |
| 修复数据对齐问题 | `fix/target-alignment` |

一个独立任务使用一个分支；不同成员不要同时推送同一个工作分支。

## 4. 模型成员主要修改什么

以接入 LSTM 为例：

| 文件 | 工作 |
|---|---|
| `src/models/lstm.py` | 新增模型及 `fit_predict` 接口 |
| `configs/aapl_lstm.json` | 复制公共配置，保留共同数据规则，增加模型参数 |
| `scripts/run_experiment.py` | 接入模型选项、训练调用和权重保存 |
| `tests/` | 对新增模型的关键行为添加必要检查 |
| `README.md` | 补充实际可执行的运行方法 |

上述 LSTM 文件和命令接入属于待完成工作。当前入口仅支持 `baselines`、`xgboost`、`patchtst`，不能仅创建模型文件就直接运行 `--models lstm`。

复用公共数据处理和评价代码。修改 `src/data.py`、`src/evaluate.py`、原始数据或共同时间划分前，先与小组同步；模型自己的参数放在独立配置中。多人需要编辑同一个实验入口时，应在 PR 中指出，便于协调。

共同实验至少保持：相同数据版本、AAPL 次日收益目标、60 日窗口、8 个特征、训练与验证日期、收益单位和评价指标。具体约定以 PROJECT_PLAN 为准。最终测试集继续封存。

## 5. 检查、提交和推送

完成修改后，运行代码检查及自己模型的实际验证实验：

```bash
python -m unittest discover -s tests -v
git status
git diff
```

代码检查通过与模型预测效果好是两回事，PR 中应分别报告。若某项验证没有执行，应说明原因。

只把本次需要提交的文件加入暂存区。例如完成上述 LSTM 接入后，根据实际修改选择：

```bash
git add src/models/lstm.py configs/aapl_lstm.json
git add scripts/run_experiment.py README.md
git diff --cached --stat
git diff --cached --name-only
git diff --cached
```

新增测试文件也应按实际文件名加入。文档任务则只加入相应文档。核对暂存区后提交：

```bash
git commit -m "Add LSTM model and experiment configuration"
git push -u origin model/lstm-xiaowang
```

之后同一分支上的修改继续 `git add`、`git commit`、`git push`，不必重复设置上游分支。不要使用强制推送处理普通的同步问题。

提交范围：

- 代码、配置、测试、共享文档和约定的数据 CSV 可以进入仓库。
- `artifacts/`、模型权重和 `private/` 按 `.gitignore` 保持本地。
- 不把整个项目的 ZIP 压缩包加入提交；它容易包含重复数据、实验输出和个人资料。
- 六份 CSV 随仓库共享，但更新数据内容时必须同步版本及原因。

## 6. 在 GitHub 创建 PR

推送后进入仓库的 Pull requests 页面，创建新的 PR：

```text
base: main
compare: model/lstm-xiaowang
```

PR 标题写清本次改动，描述可按下面填写：

```text
本次修改
- 增加了什么，解决什么问题
- 是否修改公共数据处理、评价或配置

如何运行
- 实际可执行的命令

验证情况
- 执行了哪些代码检查，结果如何
- 是否完成验证集实验，使用什么配置和随机种子
- 未验证的内容及原因

实验结果
- RMSE、MAE、方向准确率等
- 完整结果包的共享位置
- 当前结论与限制
```

请求另一位成员审查。若需要修改，继续在原分支提交并推送，同一个 PR 会自动更新，无需重新创建。

## 7. 审查与合并

审查者重点核对：

1. 数据版本、目标日期、输入窗口及真实标签是否与公共实验一致。
2. 是否使用未来信息或测试集调参。
3. 是否能按说明运行，预测是否为原始小数收益率。
4. 逐日预测、配置、训练记录和结果是否完整。
5. 是否意外修改其他模型或公共逻辑。

确认后在 GitHub 合并 PR。作者也应确认合并后的代码能与公共流程一起运行。

各成员在工作区干净后更新主分支：

```bash
git switch main
git pull --ff-only origin main
```

下一项任务从更新后的 `main` 创建新分支。

## 8. 工作未完成时，同步最新主分支

先把自己的有效修改提交在当前工作分支，再执行：

```bash
git fetch origin
git merge origin/main
```

这会将最新主分支合入当前工作分支，不会切换到 `main`。合并成功后重新运行相关检查，再执行 `git push`。

若发生冲突，Git 会指出文件，其中常出现：

```text
<<<<<<< HEAD
当前工作分支的内容
=======
主分支的内容
>>>>>>> origin/main
```

理解两边修改后编辑成最终内容，并删除这些标记。对每个解决完成的文件执行 `git add 文件名`，运行相关检查，再完成合并提交：

```bash
git commit -m "Merge latest main and resolve conflicts"
git push
```

不确定时，与另一位修改者共同确认，不直接覆盖一方。若要取消本次未完成的合并，可执行 `git merge --abort`；这也是为什么合并前应先提交自己的工作。

若 `git pull --ff-only` 或推送被拒绝，先查看 `git status` 并保留报错，确认分支情况后再处理，不执行 `--force` 或 `reset --hard` 来绕过问题。

## 9. 代码合并与实验结果合并

GitHub PR 合并代码和配置；实验结果目录通过小组约定的位置单独共享。在 PR 中提供对应结果包位置，至少包含逐日预测、指标、配置、数据与环境记录、训练记录及模型权重。

结果整合者先核对数据版本和实验约定，再核对预测目标日期与真实收益，按 `target_date` 对齐各模型，并使用公共评价代码重新计算指标。不要只收集截图，也不要静默删除不匹配的日期。

实际负责人、共享位置和最终测试安排由小组确认，并更新 PROJECT_PLAN。
