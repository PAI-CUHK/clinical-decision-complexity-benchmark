# clinical-decision-complexity-benchmark

医疗 NP 问题 LLM Benchmark

本地整理的公开版本，包含医疗问题实例生成、模型调用、验证和最终图表绘制代码。源项目的原始数据与实验结果未包含在此目录。`fuction` 保留原有拼写。

## 内容与数据范围

- `core/`：benchmark 共用的任务定义和运行框架。
- `tasks/`：按任务分目录组织实例生成、适配、渲染、验证及参考求解代码。
- `utils/`：生成和验证代码共用的工具，包括 HPO 表型相似度计算。
- `fuction/` 和 `run_all_medical_np_problems*.py`：模型调用和运行入口。
- `analysis/`：从保留的中间数据生成最终图表。
- `reports/plot_data/`：最终图表需要的六个中间数据文件，并非原始模型输出。

护士排班保留 NP-complete 任务，未包含 Nurse Scheduling NP-hard 任务。

## 任务名称

项目使用以下九个任务缩写。`(D)` 表示决策任务，`(O)` 表示优化任务：

| 缩写 | 任务 |
| --- | --- |
| RR-IAPD | RR-interval pattern detection |
| ED-TQM | Emergency triage queue management |
| ECD-RS | Early deterioration risk stratification |
| PC-MDE (D) | Multi-disease explanation decision |
| HNR (D) | Nurse rostering feasibility |
| SCS (D) | Surgical scheduling decision |
| PC-MDE (O) | Multi-disease explanation optimization |
| SCS (O) | Surgical scheduling optimization |
| PRA-CC | Patient-to-room allocation |

任务定义集中在 `core/tasks.py`。运行入口的 `--problems` 使用任务缩写；包含空格或括号的缩写需要加引号，例如：

```bash
python run_all_medical_np_problems.py --problems RR-IAPD 'PC-MDE (D)' 'SCS (O)' --generate-only
```

任务源码统一放在 `tasks/<任务缩写>/` 下，文件按职责命名，如 `generator.py`、`adapter.py`、`renderer.py`、`validator.py`、`scorer.py` 和 `baseline.py`。PC-MDE 与 SCS 的决策版和优化版共用各自的任务目录；需要区分的文件使用 `_d` 和 `_o` 后缀，例如 `tasks/pc_mde/adapter_d.py` 和 `tasks/pc_mde/adapter_o.py`。共用的目标函数和求解器放在相应任务目录内；各任务只保留实际需要的职责文件。

HNR (D) 使用随机生成的日间、晚间和夜间人员需求，不注入固定冲突模式。难度等级控制护士人数、排班天数和平均需求比例（Level 1 为 0.45，Level 10 为 0.675）；生成器支持通过 `demand_ratio`、`demand_variation` 和 `seed` 调整负载及复现结果。实例的 Yes/No 答案由已证明的最优惩罚和阈值确定；若求解器未能证明最优，实例会被标为无效并跳过。最优惩罚为零时，非负阈值下只能生成 Yes，因此不保证 Yes/No 均衡。

ED-TQM、RR-IAPD 和 SCS 生成器已调整为随机采样，不再预置原有的对抗组合或紧贴边界的模式；ECD-RS 的当前生成路径也不再调用预设陷阱。PC-MDE 的决策阈值使用可行参考解和保守上界，生成器按任务规模随机采样候选疾病及其表型，不再按目标难度筛选。已有 `reports/plot_data/` 来自旧版实验，新生成实例不能与这些历史结果视作同一分布。 PC-MDE (O) 的验证使用限时启发式搜索得到的参考解，代码无法证明该参考解是全局最优；判定和得分应按这一限制理解。

原始运行结果、生成的实例、HPO 原始及加工数据、密钥、日志和缓存均未包含。因此部分实例生成、模型运行和从头统计功能需要自行准备原始数据；仅凭本目录不能重算这些中间数据。公开前请自行确认中间数据及第三方数据的使用权限。

## 从中间数据绘图

从项目根目录执行：

```bash
python -m analysis.regenerate_all_plots --output-dir /tmp/mednp-plots
```

复现入口要求两个 Python 环境：Matplotlib 3.9.2（NumPy 1.26.4、Pillow 11.0.0）和 Matplotlib 3.11.0（NumPy 2.4.6、Pillow 12.3.0）。可分别使用 `--python-mpl39` 和 `--python-mpl311` 指定解释器。原图使用 Arimo 和 Tinos 字体；缺少字体可能影响外观。

六个 JSON 和 CSV 文件仅保留当前图表所需的数据，仅包含绘图所需的汇总数据，并不包含原始模型输出。本项目不包含从原始结果生成这些中间数据的代码。

## 模型配置

请自行配置服务与依赖。模型调用代码支持 `MEDNP_SECRETS_FILE` 和 `RESPONSES_BASE_URL` 环境变量；不要将真实密钥或本地数据加入版本库。运行入口支持 `--help` 查看参数。`vllm` 后端通过 `RESPONSES_BASE_URL` 连接 Responses API；保留的 `--vllm_port` 参数仅用于兼容旧命令，不会修改连接地址。

运行 benchmark 和分析时，按所用功能还需要安装 NumPy、SciPy、Matplotlib、pandas、OR-Tools、tqdm、requests、openai。项目尚未提供经过完整验证的统一依赖锁定文件。HPO 相关功能需要在 `data/` 下自行准备对应的 `hp.obo`、`phenotype.hpoa` 及加工文件，具体依赖取决于所运行的功能。

`analysis/` 保留从中间数据绘制图表所需的代码；其他图表、实验指标和数据导出代码不在此版本中。

项目尚未指定许可证；请由权利人选择许可证并核查依赖及数据授权。
