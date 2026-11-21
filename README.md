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

项目使用论文中的九个缩写。`(D)` 表示决策任务，`(O)` 表示优化任务：

| 缩写 | 任务 | 旧名称（结果与实例文件使用） |
| --- | --- | --- |
| RR-IAPD | RR-interval pattern detection | ECG Pattern Detection |
| ED-TQM | Emergency triage queue management | ED Scheduling |
| ECD-RS | Early deterioration risk stratification | Medical Risk |
| PC-MDE (D) | Multi-disease explanation decision | Disease Diagnosis NP-complete |
| HNR (D) | Nurse rostering feasibility | Nurse Scheduling |
| SCS (D) | Surgical scheduling decision | OR Scheduling NP-complete |
| PC-MDE (O) | Multi-disease explanation optimization | Disease Diagnosis |
| SCS (O) | Surgical scheduling optimization | OR Scheduling |
| PRA-CC | Patient-to-room allocation | PASP |

任务映射集中定义在 `core/tasks.py`。运行入口的 `--problems` 接受缩写，也接受旧名称；包含空格或括号的缩写需要加引号，例如：

```bash
python run_all_medical_np_problems.py --problems RR-IAPD 'PC-MDE (D)' 'SCS (O)' --generate-only
```

为兼容已有数据，实例目录、结果文件和分析数据中的旧标识符不改名；因此读取旧结果无需迁移。模型提示词中的任务描述也保持原样，以免更改基准输入。

任务源码统一放在 `tasks/<任务缩写>/` 下，文件按职责命名，如 `generator.py`、`adapter.py`、`renderer.py`、`validator.py`、`scorer.py` 和 `baseline.py`。PC-MDE 与 SCS 的决策版和优化版共用各自的任务目录；需要区分的文件使用 `_d` 和 `_o` 后缀，例如 `tasks/pc_mde/adapter_d.py` 和 `tasks/pc_mde/adapter_o.py`。共用的目标函数和求解器放在相应任务目录内；各任务只保留实际需要的职责文件。

外部代码若直接导入旧模块路径，需要改用 `tasks` 下的新路径；类名、函数名和历史数据键保持原样。

HNR (D) 使用随机生成的日间、晚间和夜间人员需求，不注入固定冲突模式。难度等级控制护士人数、排班天数和平均需求比例（Level 1 为 0.45，Level 10 为 0.675）；生成器支持通过 `demand_ratio`、`demand_variation` 和 `seed` 调整负载及复现结果。实例的 Yes/No 答案由已证明的最优惩罚和阈值确定；若求解器未能证明最优，实例会被标为无效并跳过。最优惩罚为零时，非负阈值下只能生成 Yes，因此不保证 Yes/No 均衡。

ED-TQM、RR-IAPD 和 SCS 生成器已调整为随机采样，不再预置原有的对抗组合或紧贴边界的模式；ECD-RS 的当前生成路径也不再调用预设陷阱。PC-MDE 的决策阈值使用可行参考解和保守上界，生成器按任务规模随机采样候选疾病及其表型，不再按目标难度筛选。已有 `reports/plot_data/` 来自旧版实验，新生成实例不能与这些历史结果视作同一分布。 PC-MDE (O) 的验证使用限时启发式搜索得到的参考解，代码无法证明该参考解是全局最优；判定和得分应按这一限制理解。

原始运行结果、生成的实例、HPO 原始及加工数据、密钥、日志和缓存均未包含。因此部分实例生成、模型运行和从头统计功能需要自行准备原始数据；仅凭本目录不能重算这些中间数据。公开前请自行确认中间数据及第三方数据的使用权限。

## 与论文的对应范围

当前公开版本保留论文中涉及的九个任务和图 2–7 所需的绘图数据与代码；图 1 是框架示意图，不由这里的绘图脚本生成。图 7 使用的是距离敏感参考分数 $S_d$ 的汇总数据，不是旧版单一参考解差距。

论文所用的完整实例、模型原始结果以及从这些结果生成中间数据的流程不在此版本中，因此不能仅凭本仓库重新计算 $S_d$、验证全部配对实例或重跑论文实验。图 7 的两个条件按各自有效答案分别汇总，源项目记录显示 PC-MDE 两种表述的实例对应关系尚未逐项验证；不应把图中的差异直接解释为已验证的配对效应。

当前生成和验证代码与论文仍有重要差别：PC-MDE 的目标函数代码与论文的部分公式和默认权重仍有差异；HNR (D) 不保证生成的 Yes/No 数量均衡，部分任务求解限时也与论文的 60 秒设置不同。现有图 2 中 Claude、DeepSeek 和 Mistral 的 NP-hard 汇总值与论文表 II 不完全一致。这些差异尚未统一，不能将当前代码或绘图结果宣称为论文实验的完整复现。

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

`analysis/` 保留论文最终图从中间数据绘制所需的代码；其他图表、实验指标和数据导出代码不在此版本中。

项目尚未指定许可证；请由权利人选择许可证并核查依赖及数据授权。
