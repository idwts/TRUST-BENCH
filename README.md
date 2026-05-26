# TRUST-BENCH

This repository contains the code and data for **Trust No Tool: Evaluating and Defending LLM Agents under Untrusted Tool Feedback**.

## Repository Structure

```text
TRUST-BENCH/
├── README.md
├── data/
│   ├── clean_schema.json
│   ├── episodes_paper_id_v3.json
│   ├── folds_paper_id_v3.json
│   └── episodes_external_ood_v2.json
└── code/
    ├── risk_utils.py
    ├── train_risk_model_full.py
    ├── run_grouped_cv.py
    ├── aggregate_fold_metrics.py
    └── state_risk_coefficient_sensitivity.py
```

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@article{yan2026trust,
  title={Trust No Tool: Evaluating and Defending LLM Agents under Untrusted Tool Feedback},
  author={Yan, Lecheng and Li, Ruizhe and Han, Xicheng and Li, Wenxi and Wang, Binwu and Wang, Longyue and Lyu, Chenyang and Chen, Guanhua},
  journal={arXiv preprint arXiv:2605.17453},
  year={2026}
}
```

**Paper:** https://arxiv.org/abs/2605.17453
