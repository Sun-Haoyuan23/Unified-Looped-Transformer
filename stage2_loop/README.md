# 第二阶段交付

- [实验结果与失败诊断](STAGE2_REPORT.md)
- [下一轮可执行方案](NEXT_EXPERIMENT.md)
- [文献定位与研究风险](RESEARCH_DECISIONS.md)
- [共享循环生成对照图](elt_contact_sheet.jpg)
- [原权重截层生成对照图](truncate_contact_sheet.jpg)
- [训练汇总](train_summary.json)、[循环评估](eval_summary.json)、[截层评估](truncate_eval_summary.json)、[循环 PickScore](pickscore_summary.json)、[截层 PickScore](truncate_pickscore_summary.json)
- `elt_tied_tail_rank16.pt` 是**失败试测**的适配器权重，仅供复核负结果；`logs/` 含逐样本耗时/显存记录。

源程序：`train_elt.py`、`run_elt_infer.py`、`run_head_truncate.py`、`launch_loop.py`、`evaluate_elt.py`、`evaluate_pickscore.py`。所有结果基于 Stage 1 的官方 Show-o2 1.5B 权重和同一组提示/种子。
