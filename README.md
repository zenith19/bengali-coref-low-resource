# Bengali Coreference Resolution under Low-Resource Constraints

This repository contains the experimental code for the master's thesis:

> **Developing Deep Learning Models for Coreference Resolution in Bengali**
>
> Zenith Biswas, Saarland University, 2026
>
> Supervised by Andrew Dyer (M.A.) · Referees: Prof. Dr. Annemarie Verkerk, Prof. Dr. Michael Hahn

It also accompanies the peer-reviewed paper:

> **Quality over Quantity: Rethinking Data and Model Assumptions for Bengali Coreference Resolution**
>
> Zenith Biswas and Andrew Thomas Dyer. KONVENS 2026 (oral presentation).

## Overview

This thesis systematically investigates transformer-based approaches to Bengali coreference resolution — the task of identifying all expressions that refer to the same entity. Bengali, despite having 250 million speakers, has very limited NLP resources for this task.

**Key findings:**
- **Best result:** MuRIL-Large achieves **66.73% CoNLL F1** with a frozen encoder architecture
- **Quality > Quantity:** 41 gold-standard documents outperform 141 mixed-quality documents
- **Augmentation doesn't help:** Back-translation and paraphrasing provide no benefit and can degrade performance
- **Cross-lingual transfer is limited:** Hindi transfer is neutral; English transfer trends negative
- **Embedding collapse:** Several widely-used multilingual models (XLM-RoBERTa variants, MuRIL-Base) produce collapsed embedding spaces for Bengali

## Architecture

All experiments use a **frozen encoder** approach:

1. **Frozen pre-trained transformer** extracts span embeddings `[h_start; h_end; h_mean]` (2304–3456 dim depending on encoder)
2. **Pairwise features** are constructed by concatenating embeddings with element-wise product and absolute difference (9216–13824 dim)
3. **Trainable MLP classifier** (`[12·d] → 512 → 256 → 1`, ~4.85M parameters for 768-dim encoders, ~6.4M for 1024-dim, ~7.2M for 1152-dim) predicts coreference scores
4. **Graph-based connected components** clustering forms final coreference chains

## Models

| Model | Category | HuggingFace ID |
|-------|----------|----------------|
| MuRIL-Large | Indic-focused | `google/muril-large-cased` |
| mBERT | Multilingual | `bert-base-multilingual-cased` |
| RemBERT | Multilingual | `google/rembert` |
| BanglaBERT-Base | Bengali-specific | `csebuetnlp/banglabert` |
| BERT-Base-Uncased | Control baseline | `bert-base-uncased` |

## Repository Structure

```
├── data/
│   └── dataset_split_info.json          # Train/dev/test document IDs, stratification, random seed
│
├── notebooks/
│   ├── data_preparation/
│   │   ├── making_datasets.ipynb            # Stratified train/dev/test splitting of BenCoref
│   │   └── making_augmented_datasets.ipynb  # Back-translation & paraphrase augmentation
│   │
│   ├── preliminary_study/
│   │   ├── model_selection.ipynb            # Stage 1: 10 model evaluation on full BenCoref
│   │   ├── clustering_comparison.ipynb      # Comparison of 4 clustering algorithms
│   │   └── training_strategy_comparison.ipynb  # Frozen vs LoRA vs full fine-tuning
│   │
│   ├── experiments/
│   │   ├── exp0_zero_shot_baseline.ipynb    # No training, pre-trained representations only
│   │   ├── exp1_mixed_baseline.ipynb        # Fine-tuning on 141 mixed Bengali docs
│   │   ├── exp2_full_backtranslation.ipynb  # + Back-translation augmentation (282 docs)
│   │   ├── exp3_full_paraphrase.ipynb       # + Paraphrase augmentation (282 docs)
│   │   ├── exp4_hindi_transfer.ipynb        # Cross-lingual from Hindi (80 docs)
│   │   ├── exp5_english_transfer.ipynb      # Cross-lingual from English (437 docs)
│   │   ├── exp6_bencoref_only.ipynb         # Gold-standard only (41 docs)
│   │   ├── exp7_bencoref_backtranslation.ipynb  # Gold + BT augmentation (82 docs)
│   │   └── exp8_bencoref_paraphrase.ipynb   # Gold + paraphrase augmentation (82 docs)
│   │
│   └── analysis/
│       ├── comparison_analysis.ipynb        # Cross-experiment results comparison
│       ├── statistical_significance.ipynb   # Paired bootstrap & Holm–Bonferroni correction (corrected*)
│       └── error_analysis.ipynb             # Error analysis on best model (MuRIL-Large, Exp 6)
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Experiments

The experiments follow a two-stage design: a preliminary study for model selection (10 models), followed by nine main experiments (E0–E8) on a held-out test set.

| Exp | Notebook | Description | Training Data | Docs |
|-----|----------|-------------|---------------|------|
| 0 | `exp0_zero_shot_baseline` | Zero-shot (no training) | None | 0 |
| 1 | `exp1_mixed_baseline` | Fine-tuning on all Bengali data | 100 TransMuCoRes + 41 BenCoref | 141 |
| 2 | `exp2_full_backtranslation` | + Back-translation augmentation | 141 original + 141 BT | 282 |
| 3 | `exp3_full_paraphrase` | + Paraphrase augmentation | 141 original + 141 Para | 282 |
| 4 | `exp4_hindi_transfer` | Cross-lingual from Hindi | 80 Hindi TransMuCoRes | 80 |
| 5 | `exp5_english_transfer` | Cross-lingual from English | 437 English OntoNotes | 437 |
| 6 | `exp6_bencoref_only` | Gold-standard only (ablation) | 41 BenCoref | 41 |
| 7 | `exp7_bencoref_backtranslation` | Gold + BT augmentation | 41 BenCoref + 41 BT | 82 |
| 8 | `exp8_bencoref_paraphrase` | Gold + paraphrase augmentation | 41 BenCoref + 41 Para | 82 |

All experiments share the same dev (10 docs) and test (71 docs) splits from BenCoref.

## Datasets

- **[BenCoref](https://github.com/Bangla-NLP/BenCoref)** — 122 gold-standard Bengali coreference documents
- **[TransMuCoRes](https://github.com/ritwikmishra/transmucores)** — Silver-standard Bengali/Hindi data via NLLB translation + awesome-align projection from English LitBank
- **[OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)** — English gold-standard coreference (accessed via HuggingFace `conll2012_ontonotesv5`)

### Reproducing the Data Splits

The original datasets are not redistributed here. To reproduce the exact train/dev/test splits used in all experiments:

1. Download BenCoref from the [official repository](https://github.com/Bangla-NLP/BenCoref)
2. Download TransMuCoRes from the [official repository](https://github.com/ritwikmishra/transmucores)
3. Use `data/dataset_split_info.json` which contains the exact document IDs for each split (random seed: 42), stratification details by domain, and training data composition for all experiments
4. Run `notebooks/data_preparation/making_datasets.ipynb` to generate the split `.conll` files
5. Run `notebooks/data_preparation/making_augmented_datasets.ipynb` to generate the augmented training sets

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: Python 3.12, PyTorch 2.9+, Transformers 4.x, scikit-learn, NetworkX, SciPy. See `requirements.txt` for full list.

**Hardware used:** NVIDIA L4 GPU (24 GB VRAM), 8-core Intel Xeon, 32 GB RAM.

## Evaluation Metrics

All experiments report **CoNLL F1**, the average of three standard coreference metrics:
- **MUC** — Link-based (how many links need to be added/removed)
- **B³** — Mention-based (per-mention precision and recall)
- **CEAFₑ** — Entity-based (optimal alignment between predicted and gold clusters)

## Post-Submission Correction

\* The statistical significance analysis (`statistical_significance.ipynb`) has been corrected after thesis submission. The original analysis incorrectly pooled per-document scores across all five models in the paired bootstrap test, violating independence assumptions. The corrected version applies Holm–Bonferroni correction per comparison family: aggregate tests corrected across 7 comparisons, per-model tests corrected within each group. This strengthened the gold vs. mixed finding (p = 0.0070, previously p = 0.0400) and revealed a newly significant result that back-translation on gold data degrades performance (p = 0.0156).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{biswas2026quality,
  title={Quality over Quantity: Rethinking Data and Model Assumptions for Bengali Coreference Resolution},
  author={Biswas, Zenith and Dyer, Andrew Thomas},
  booktitle={Proceedings of KONVENS 2026},
  year={2026}
}
```

The repository is also based on the underlying thesis:

```bibtex
@mastersthesis{biswas2026bengali,
  title={Developing Deep Learning Models for Coreference Resolution in Bengali},
  author={Biswas, Zenith},
  year={2026},
  school={Saarland University},
  type={Master's Thesis}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
