# Bengali Coreference Resolution under Low-Resource Constraints

Bengali has over 250 million speakers and one gold-standard coreference dataset. This repository tests what actually happens when you apply the standard low-resource toolkit (more data, augmentation, cross-lingual transfer, multilingual encoders) on a span-sensitive task. Three common assumptions did not survive:

- **41 gold-standard documents outperform 141 mixed gold + silver ones**, for every model tested.
- **Back-translation degrades performance**, because span labels do not survive sentence rewrites.
- **Three of ten widely-used multilingual encoders are collapsed for Bengali**, so coreferent and non-coreferent mention pairs are indistinguishable.

Best result: **66.73 CoNLL F1** (MuRIL-Large, frozen encoder, 41 gold training documents).

📄 **Paper:** Quality over Quantity: Rethinking Data and Model Assumptions for Bengali Coreference Resolution. Zenith Biswas and Andrew Thomas Dyer. KONVENS 2026 (oral presentation).
🎓 **Thesis:** Developing Deep Learning Models for Coreference Resolution in Bengali. Saarland University, 2026. Supervised by Andrew Dyer (M.A.); referees Prof. Dr. Annemarie Verkerk and Prof. Dr. Michael Hahn.

## Results

### Quality beats quantity

CoNLL F1 (%) on 71 held-out gold BenCoref test documents. Gold-only training uses 41 documents; mixed adds 100 silver TransMuCoRes-BN documents.

| Model | Zero-shot | Mixed (141 docs) | **Gold only (41 docs)** |
|---|---|---|---|
| MuRIL-Large | 62.44 | 66.70 | **66.73** |
| mBERT | 63.61 | 62.33 | **64.15** |
| RemBERT | 63.20 | 62.80 | **63.88** |
| BanglaBERT-Base | 62.38 | 63.08 | **63.39** |
| BERT-Base-Uncased | 63.82 | 63.06 | **63.64** |
| **Average** | 63.09 | 63.59 | **64.36** |

Gold-only is higher for all five models. For three of them, adding silver data scores *below* zero-shot. Per-document paired bootstrap (10,000 resamples): **+2.63 pp, 95% CI [+0.94, +4.54], p = 0.0070** after Holm–Bonferroni correction across seven comparisons.

### Augmentation and cross-lingual transfer

Corpus-level CoNLL F1 (%), averaged over the five models, on the same 71 gold test documents.

| Configuration | Training docs | CoNLL F1 |
|---|---|---|
| Zero-shot | 0 | 63.09 |
| **Gold only** | 41 | **64.36** |
| Gold + back-translation | 82 | 63.54 |
| Gold + MLM paraphrase | 82 | 63.89 |
| Mixed | 141 | 63.59 |
| Mixed + back-translation | 282 | 63.51 |
| Mixed + MLM paraphrase | 282 | 63.89 |
| Hindi transfer | 80 | 63.12 |
| English transfer | 437 | 61.67 |

### Statistical tests

Per-document paired bootstrap (10,000 resamples, two-sided) over 71 documents and five models, with Holm–Bonferroni correction across the seven aggregate comparisons. Note that the baseline differs by row, and that Δ here is the mean of per-document differences, which is not the same quantity as the corpus-level difference in the table above.

| Comparison | Baseline | Δ F1 (pp) | 95% CI | p (corrected) |
|---|---|---|---|---|
| Gold only vs mixed | Mixed | **+2.63** | [+0.94, +4.54] | **0.0070** |
| Back-translation on gold | Gold only | **−0.81** | [−1.41, −0.25] | **0.0156** |
| Back-translation on mixed | Mixed | +1.05 | [+0.09, +2.06] | 0.1470 |
| Paraphrase on mixed | Mixed | +0.83 | [+0.05, +1.65] | 0.1512 |
| Paraphrase on gold | Gold only | −0.46 | [−0.97, +0.04] | 0.2076 |
| Hindi transfer | Mixed | +0.99 | [−0.24, +2.28] | 0.2328 |
| English transfer | Mixed | +0.92 | [−0.83, +2.85] | 0.3260 |

Two of seven comparisons survive correction. Gold-only training beats mixed training, consistently across all five models, and back-translation on gold data degrades performance, driven mainly by MuRIL-Large (Δ = −2.73 pp, p = 0.0060 within that family). Back-translation reorders and rewrites sentences while positional span labels stay put, so mention boundaries no longer align with the right tokens.

Neither cross-lingual transfer setting shows a reliable effect. English transfer uses the largest training set of any configuration, 437 documents, and buys nothing: it is the weakest configuration at corpus level, while the per-document test puts it slightly above the mixed baseline with a confidence interval spanning zero. The two measures disagree in direction, which is itself the finding, and with 71 test documents the study has limited power to resolve effects of around one point.

### Embedding collapse

Discriminability gap Δ = mean cosine similarity of coreferent mention pairs − mean cosine similarity of non-coreferent pairs, on frozen embeddings, before any training.

| Model | Δ | | Model | Δ |
|---|---|---|---|---|
| Bangla-BERT-Base | 0.121 | | BanglaBERT-Base | 0.034 |
| RemBERT | 0.096 | | BanglaBERT-Large | 0.018 |
| MuRIL-Large | 0.086 | | XLM-RoBERTa-Large | **0.006** |
| mBERT | 0.067 | | XLM-RoBERTa-Base | **0.005** |
| BERT-Base-Uncased | 0.046 | | MuRIL-Base | **0.004** |

Bold = collapsed (Δ < 0.01). In these models every mention pair scores above 0.94 similarity regardless of whether the mentions corefer, so no threshold can separate them. MuRIL-Base collapses while MuRIL-Large does not, on the same pre-training data. The check needs no fine-tuning and no task-specific training, so it is worth running before committing compute to any similarity-based method in a new language.

## Architecture

All experiments use a **frozen encoder** approach:

1. **Frozen pre-trained transformer** extracts span embeddings `[h_start; h_end; h_mean]` (2304–3456 dim depending on encoder)
2. **Pairwise features** are constructed by concatenating embeddings with element-wise product and absolute difference (9216–13824 dim)
3. **Trainable MLP classifier** (`[12·d] → 512 → 256 → 1`, ~4.85M parameters for 768-dim encoders, ~6.4M for 1024-dim, ~7.2M for 1152-dim) predicts coreference scores
4. **Graph-based connected components** clustering forms final coreference chains

Only the classifier is trained. A preliminary pilot found frozen encoders more stable than LoRA or full fine-tuning at this data scale, at a fraction of the cost. Gold mention boundaries are assumed throughout: this is a study of coreference linking, not mention detection.

## Models

| Model | Category | HuggingFace ID |
|-------|----------|----------------|
| MuRIL-Large | Indic-focused | `google/muril-large-cased` |
| mBERT | Multilingual | `bert-base-multilingual-cased` |
| RemBERT | Multilingual | `google/rembert` |
| BanglaBERT-Base | Bengali-specific | `csebuetnlp/banglabert` |
| BERT-Base-Uncased | Control baseline | `bert-base-uncased` |

These five were selected from ten candidates in the preliminary study; three were excluded for collapsed embeddings and two as redundant. See `notebooks/preliminary_study/model_selection.ipynb`.

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

All experiments share the same dev (10 docs) and test (71 docs) splits from BenCoref, so differences between configurations come from the training data alone.

## Datasets

- **[BenCoref](https://github.com/Bangla-NLP/BenCoref)**: 122 gold-standard Bengali coreference documents
- **[TransMuCoRes](https://github.com/ritwikmishra/transmucores)**: Silver-standard Bengali/Hindi data via NLLB translation + awesome-align projection from English LitBank
- **[OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)**: English gold-standard coreference (accessed via HuggingFace `conll2012_ontonotesv5`)

BenCoref is native Bengali text; TransMuCoRes is translated English fiction. Genre and source language therefore differ alongside annotation quality, so the gold-vs-mixed comparison should not be read as isolating annotation quality alone. No cleaner silver set exists for Bengali.

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
- **MUC**: Link-based (how many links need to be added/removed)
- **B³**: Mention-based (per-mention precision and recall)
- **CEAFₑ**: Entity-based (optimal alignment between predicted and gold clusters)

MUC is largely insensitive to merged entities, so it stays high under over-linking while CEAFₑ drops sharply. In our results MUC ≈ 96 and CEAFₑ ≈ 30 across models, the signature of systematic over-linking. CoNLL F1 is reported for comparability with prior work, but conclusions are drawn from relative comparisons, which are measured identically across configurations.

## Limitations

- **One language.** Findings may extend to other low-resource, especially Indo-Aryan, settings, but this is not verified here.
- **Gold mention boundaries assumed.** End-to-end systems that also detect mentions may behave differently.
- **One architecture.** Frozen encoder plus pairwise MLP. Fine-tuned or alternative architectures may respond differently to data quality, augmentation, and transfer.
- **Small test set.** 71 documents limits statistical power; this is the likely reason the transfer effects did not reach significance.
- **Two augmentation methods.** Both sentence-level. Span-preserving or embedding-level augmentation may behave differently.

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
