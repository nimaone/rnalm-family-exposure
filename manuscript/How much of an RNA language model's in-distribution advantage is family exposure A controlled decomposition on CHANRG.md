# How much of an RNA language model's in-distribution advantage is family exposure? A controlled decomposition on CHANRG

Oct 15, 2026 · @nima

Nima Baselizadeh¹

¹ Bu-Ali Sina University, Hamedan, Iran. Correspondence: nimabaselizadeh@gmail.com

## Abstract

Pretrained RNA language models (RNA-LMs) are reported to generalize poorly to RNA families absent from training, but it is unclear how much of their in-distribution advantage reflects pretrained knowledge and how much reflects exposure to the same family during downstream training. Using the CHANRG benchmark (50–200 nt), we decompose the contribution of a frozen RNA-LM (RiNALMo-giga) paired with a fixed lightweight pair-map decoder, using a no-LM baseline, a random-weight LM control, per-family training caps and size-matched random subsets. Our RNAfold pipeline matched published length-matched baselines to within 0.014 F1. The LM improved in-distribution (ID) macro-F1 by +0.18 (95% CI 0.09–0.24) but out-of-distribution (OOD) F1 by only +0.04 (0.01–0.07); the difference (+0.14, 0.04–0.21) held across three seeds. A random-weight LM gave no benefit (−0.03 on the ID test set), indicating that the gain depends on pretrained knowledge rather than input dimensionality. The ID gain increased with the number of training examples of the same family (Spearman ρ = 0.60) and was concentrated in a few large families: capping each family at 10 training sequences removed about 60% of the ID gain (cap minus size-matched random: −0.10, CI −0.14 to −0.04; two seeds, exploratory). Weighting families equally reduced the apparent ID gain to +0.04. We recommend family-stratified and family-equal reporting. We did not reproduce the OOD collapse reported for fully fine-tuned models, and we discuss why.

**Keywords:** RNA secondary structure; language models; out-of-distribution generalization; benchmarking; family exposure.

## 1. Introduction

RNA foundation models promise to transfer knowledge learned from large unlabeled sequence collections to structure prediction, yet the evidence on how far that transfer reaches is conflicting. RiNALMo reports strong generalization to families unseen during fine-tuning (Penić et al., 2025), whereas the CHANRG benchmark reports that the same class of models collapses on families, clans and architectures that are disjoint from training (arXiv:2603.22330). Earlier work on deep-learning structure predictors had already shown weak transfer across RNA families (Szikszai et al., 2022).

These reports share an implicit reading: the in-distribution (ID) score is treated as the model's "true" ability, and the OOD score as the loss of it. That reading hides a question. ID test sequences come from families that also appear, often many times, in the training set. How much of the ID advantage of an RNA-LM is pretrained knowledge, and how much is the downstream decoder having seen the same family? The answer decides whether the ID–OOD gap measures a failure of generalization or an inflated ID baseline.

We address this with a controlled decomposition on CHANRG. We hold a lightweight pair-map decoder fixed and change one factor at a time: the LM (RiNALMo-giga, frozen), no LM, a random-weight LM, per-family training caps, and size-matched random subsets. Our contributions are:

1. **A literature-anchored pipeline.** Our RNAfold results match published length-matched numbers on all four evaluation splits to within 0.014 F1, so differences between arms are not artefacts of data handling.
2. **A causal split between pretrained knowledge and family exposure.** A random-weight LM control isolates pretrained knowledge; per-family caps and size-matched subsets isolate exposure.
3. **A composition effect in the test set.** About half of the ID gain comes from a handful of very large families, so sequence-weighted ID scores overstate the typical family.
4. **A reporting recommendation.** We recommend reporting F1 per family-size tier and with equal weight per family, next to the usual sequence-weighted score.

## 2. Related work

**CHANRG.** The CHANRG benchmark (arXiv:2603.22330) defines family-, clan- and architecture-disjoint splits and reports that fully fine-tuned RNA foundation models lose most of their ID performance OOD. We use its splits and published length-matched baselines as our reference point. Our difference: we do not ask whether the OOD drop exists, we ask how much of the ID score is exposure to the same family, and we use a frozen backbone so that the decoder is the only learned component.

**RiNALMo.** RiNALMo is a large RNA language model reported to generalize across families after fine-tuning (Penić et al., 2025); the authors also report an exception for telomerase RNAs. We use RiNALMo-giga (651M parameters, 1280-dim embeddings) as a frozen feature extractor. Our difference: we test whether its apparent ID advantage survives controls for exposure.

**Cross-family generalization.** Szikszai et al. (2022) showed that learned structure predictors generalize poorly across RNA families (a convolutional model's F1 fell from 0.72 under k-fold to 0.50 under family-fold cross-validation). Our difference: we quantify the ID side of that gap.

**RNA-LM benchmarks.** A benchmark of RNA language models for structure and function tasks (arXiv:2410.16212) compares backbones under a common protocol (Zablocki et al.; generalization stayed difficult when test sequences were dissimilar to the training data). Our difference: we study one backbone in depth, with intervention controls rather than a model comparison.

**Shifu.** Shifu (Galvez and Vicens, bioRxiv 2026) proposes a leakage-audited corpus with family-aware splits, a multi-criterion scoring scheme and compact RNA language models, and argues that part of the apparent strength of pretrained RNA-LMs reflects overlap with the pretraining distribution. Our difference: Shifu changes the corpora and the models, whereas we keep one backbone and one benchmark fixed and intervene on the family composition of the downstream training set, so our question concerns exposure during downstream training rather than during pretraining.

## 3. Methods

### 3.1 Data

We use CHANRG (gated; accessed with a Hugging Face token) restricted to sequences of 50–200 nt whose sequence and structure strings have equal length (128,814 of 170,083 sequences). We subsample 12,000 training sequences (from 96,262 eligible; 3,484 families), 300 validation sequences, and the evaluation subsets below. We verified at three levels (family, clan, architecture) and by exact sequence identity that GenA, GenC and GenF share no family with the training subset and that no evaluation sequence is identical to a training sequence.

| Split | Sequences | Families | Resampling unit (n units) |
| --- | --- | --- | --- |
| Test (ID) | 1000 | 224 | family (224) |
| GenA | 1000 | 102 | super-family (99) |
| GenC | 1000 | 28 | clan (11) |
| GenF | 800 | 10 | clan (10) |

The resampling unit is the coarsest level at which each split is disjoint from training. GenC and GenF have only 11 and 10 units, so their confidence intervals are wide; we report them but do not rely on them.

### 3.2 Models (arms)

All learned arms share one decoder: a U-Net (width 32, 583K parameters) that maps pairwise features to a base-pair probability map. Inputs differ by arm.

- **unet+phys (main):** frozen RiNALMo-giga embeddings (1280-d) plus physics features from ViennaRNA 2.7.2.
- **unet:** the same, without physics features.
- **nolm+phys:** physics features only.
- **randlm+phys:** embeddings from a randomly initialised RiNALMo-giga (same architecture, seed 0), plus physics features. This controls for input dimensionality.
- **Reference methods:** RNAfold minimum-free-energy (MFE) and RNAfold base-pair-probability thresholding (threshold 0.75).
- **LoRA / adapter (supplement):** LoRA on 8 modules in layers 29–32 of RiNALMo (163.8K parameters), a parameter-matched rank-64 adapter, and a frozen control, all trained for 1,500 steps with two accumulation steps.

Training: 3 epochs (36,000 updates at N = 12,000), mixed precision, one NVIDIA T4 GPU, seeds 0, 1, 2. The decision threshold is chosen on validation. A second frozen backbone, RNA-FM (99.5M parameters, 12 layers, 1280-dim embeddings; Chen et al., 2022), loaded through MultiMolecule in the same way, was run for the main comparison (b2+phys, three seeds).

&#91;image: Figure 1\]

*Figure 1. Experimental design. Three input arms and the RNAfold baseline share one decoder; three training-set interventions change only the training data.*

### 3.3 Interventions

- **Per-family cap.** At most 10 training sequences per family (8,924 of 12,000 retained; 73 families exceeded the cap; 3,076 sequences removed). Two seeds.
- **Size-matched random subset.** A random subset of the same size (N = 8,924), which separates the effect of a smaller training set from the effect of the cap.
- **Scale curve.** Nested family-random training subsets of 1,500, 3,000, 6,000 and 12,000 sequences.

### 3.4 Metrics

We report macro-F1 (mean of per-sequence F1) and micro-F1 (pooled base pairs). The tables use macro-F1 unless stated. "Gain" means the difference in F1 between two arms. "Exposure" of a test sequence is the number of training sequences from its family. "Retention" is mean OOD F1 divided by ID F1.

### 3.5 Statistics

Confidence intervals come from a bootstrap that resamples resampling units and seeds jointly (4,000 resamples). Confirmatory tests (36 in total, listed in the Supplement) are corrected with the Holm procedure and require the same sign in every seed. The analyses of exposure (Section 4.4), of the cap (Section 4.5) and of the weighting scheme (Section 4.6), and the scale curve, were added after seeing the main results and are **exploratory**: they are not part of the Holm family.

## 4. Results

### 4.1 Our RNAfold pipeline matches the literature

Our RNAfold-MFE macro-F1 differs from the published length-matched values by at most 0.007 on every split, and micro-F1 by at most 0.014 (Table 1; Figure 1 shows the experimental design). The published numbers use the full splits and ours use subsets, so this shows that the data handling is comparable, not that the protocols are identical.

| Split | RNAfold-MFE, this work (macro-F1) | RNAfold, published (length-matched) |
| --- | --- | --- |
| Test | 0.319 | 0.320 |
| GenA | 0.427 | 0.419 |
| GenC | 0.279 | 0.280 |
| GenF | 0.280 | 0.280 |

*Table 1. RNAfold anchor against published length-matched baselines (50–200 nt).*

### 4.2 The LM gain is large in distribution and small out of distribution

Adding frozen RiNALMo-giga to the decoder raised ID macro-F1 by +0.182 (95% CI 0.088–0.246; Holm-adjusted p = 0.018; all three seeds positive) and mean OOD macro-F1 by +0.044 (0.007–0.069). The difference, +0.139 (0.043–0.213), is positive in all three seeds (Table 2, Figure 2). Per OOD split, the interaction is clear for GenC (+0.166, 0.073–0.245) and GenF (+0.162, 0.067–0.229) but not for GenA (+0.089, −0.038 to 0.207), where the LM gain is larger (+0.093) and its interval includes zero. GenC and GenF have only 11 and 10 resampling units.

| Arm | Test | GenA | GenC | GenF | OOD mean | Retention (OOD / Test) |
| --- | --- | --- | --- | --- | --- | --- |
| unet+phys (LM) | 0.546 | 0.533 | 0.302 | 0.309 | 0.381 | 0.70 (0.55–0.99) |
| unet (LM, no physics) | 0.552 | 0.502 | 0.274 | 0.317 | 0.364 | 0.66 (0.53–0.93) |
| nolm+phys | 0.363 | 0.440 | 0.286 | 0.288 | 0.338 | 0.93 (0.69–1.38) |
| randlm+phys | 0.336 | 0.428 | 0.270 | 0.285 | 0.328 | 0.98 (0.71–1.48) |
| RNAfold-MFE | 0.319 | 0.427 | 0.279 | 0.280 | 0.329 | 1.03 (0.74–1.56) |

*Table 2. Macro-F1 per arm, mean of three seeds. Retention = mean OOD F1 / Test F1, with 95% CI.*

&#91;image: Figure 2\]

*Figure 2. (A) Macro-F1 per arm and split (three seeds). (B) Difference between the LM gain on Test and the LM gain on each OOD split, with 95% bootstrap CI.*

Without the LM, the decoder with physics features keeps almost all of its ID score OOD (retention 0.93), and it beats RNAfold-MFE by +0.045 (0.018–0.070) on Test and by +0.006 to +0.013 on the OOD splits, which is not significant after correction. So the ID–OOD gap in our setting is created by the LM arm.

### 4.3 The gain depends on pretrained weights

Replacing the pretrained LM with a random-weight LM of the same architecture removed the gain: pretrained minus random-weight LM was +0.210 on Test (0.102–0.285; Holm-adjusted p = 0.018) and +0.054 OOD (0.016–0.079). The random-weight LM was slightly worse than no LM at all (−0.028 on Test, −0.047 to −0.010; Holm-adjusted p = 0.018), so high-dimensional extra input does not explain the gain (Figure 3).

&#91;image: Figure 3\]

*Figure 3. Pretrained LM, random-weight LM and no LM on Test and on the mean of the three OOD splits (macro-F1, three seeds).*

### 4.4 The ID gain grows with exposure to the same family (exploratory)

We grouped Test sequences by how many training sequences come from their family and computed the LM gain per group (Table 3, Figure 4). Families with 1–9 training sequences gained +0.010 (−0.004 to 0.023), no more than the OOD gain of +0.044. Families with 50 or more training sequences gained +0.236 to +0.299. Across the 56 families with at least three test sequences, the gain correlated with log10 of the family's training count (Spearman ρ = 0.60, 95% CI 0.39–0.75).

| Training sequences in the family | Test sequences | Families | LM gain (95% CI) |
| --- | --- | --- | --- |
| 1–9 | 206 | 149 | +0.010 (−0.004 to 0.023) |
| 10–49 | 216 | 64 | +0.076 (0.043 to 0.109) |
| 50–199 | 138 | 7 | +0.236 (0.161 to 0.280) |
| ≥200 | 440 | 4 | +0.299 (0.176 to 0.331) |
| 0 (OOD, pooled) | – | – | +0.044 (0.007 to 0.069) |

*Table 3. LM gain (unet+phys minus nolm+phys, macro-F1) by family exposure. The ≥200 row has only four families, so its interval is not reliable.*

Three further checks point the same way. Removing the single largest test family (271 of 1,000 sequences) lowers the Test gain from +0.183 to +0.127. Removing the five largest lowers it to +0.072, and the ID–OOD interaction is then no longer distinguishable from zero (+0.028, −0.009 to 0.073). In the scale curve, the Test gain rose from +0.131 at 1,500 training sequences to +0.183 at 12,000, while the OOD gain rose from +0.023 to +0.044, and the slope per doubling was larger for Test (difference +0.011, 0.003–0.019). The relation is between families, not shown within a family: a regression with family fixed effects gave +0.076 per log10 exposure with an interval that includes zero (−0.001 to 0.150; p = 0.058).

&#91;image: Figure 4\]

*Figure 4. (A) LM gain by number of same-family training sequences, with 95% CI; the ≥200 group has four families. (B) LM gain on Test and on OOD as the training set grows. Exploratory.*

### 4.5 Capping each family removes about 60% of the ID gain (exploratory)

We capped every family at 10 training sequences (N = 8,924) and compared against a random subset of the same size. Test F1 of the LM arm fell from 0.523 (random subset) to 0.401 (cap). The LM gain on Test fell from +0.172 to +0.073, a difference of −0.099 (−0.145 to −0.036). The gain on OOD changed little (−0.013, −0.022 to −0.003), and the random subset did not differ from the full training set (−0.004, −0.014 to 0.004), so the loss comes from the cap and not from having fewer sequences (Table 4, Figure 5).

| Family size in training | Families | Test sequences | Gain, full | Gain, random subset | Gain, cap | Cap minus random (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| <10 | 11 | 36 | +0.006 | +0.009 | +0.017 | +0.008 (−0.010 to 0.027) |
| 10–49 | 34 | 170 | +0.079 | +0.072 | +0.054 | −0.018 (−0.035 to 0.000) |
| ≥50 | 11 | 578 | +0.275 | +0.270 | +0.103 | −0.167 (−0.197 to −0.104) |

*Table 4. LM gain by family-size tier under three training conditions (macro-F1; two seeds; 56 families with at least three test sequences).*

The effect is concentrated in the large families (cap minus random −0.167 for families with 50 or more training sequences), and for the largest families the capped gain was lower than the random-subset gain in both seeds. Two caveats apply. The cap analysis uses two seeds, and the ≥50 tier contains 11 families. And the gain did not fall to the OOD level: under the cap the LM still adds +0.073 on Test, with an interaction against OOD of +0.047 (0.008–0.081). The cap removes exposure but not the other properties of large families (for example, being easy to predict). Compared with test families that have 10–19 training sequences in the full set (+0.049), the capped large families are still higher by +0.054 (−0.004 to 0.096; p = 0.057).

&#91;image: Figure 5\]

*Figure 5. LM gain on Test by family-size tier under the full training set, a size-matched random subset and the per-family cap. Annotations give cap minus random with 95% CI (two seeds). Exploratory.*

### 4.6 Weighting families equally shrinks the ID gain (exploratory)

The usual score weights every sequence equally, so large families dominate. Weighting every family (or OOD unit) equally gave an ID gain of +0.036 (0.022–0.050) instead of +0.183, and an OOD gain of −0.015 (−0.031 to −0.002) instead of +0.044 (Table 5, Figure 6). The pretrained-minus-random-weight difference behaves the same way (+0.046 on Test, −0.013 OOD).

| Weighting | Test gain | OOD gain | ID–OOD difference |
| --- | --- | --- | --- |
| Per sequence | +0.183 (0.089–0.241) | +0.044 (0.007–0.069) | +0.139 (0.042–0.211) |
| Per family or unit | +0.036 (0.022–0.050) | −0.015 (−0.031 to −0.002) | +0.051 (0.032–0.072) |

*Table 5. LM gain (macro-F1) under two weightings, three seeds.*

&#91;image: Figure 6\]

*Figure 6. (A) LM gain under per-sequence and per-family weighting. (B) LM gain on Test after removing the k largest training families from the test set (n = remaining test sequences); dashed line, OOD gain. Exploratory.*

The same tier-level view applies to F1 itself (Table 6). Without an LM, F1 falls as the family's training count grows (0.475 for families with fewer than 10 training sequences, 0.315 for families with 50 or more), whereas with RiNALMo it rises (0.485 and 0.598). The large families are therefore the hardest for the physics-based baseline and the easiest for the LM arm, so part of their large gain reflects a low baseline and not exposure alone.

| Arm | <10 (206 seq.) | 10–49 (216 seq.) | ≥50 (578 seq.) |
| --- | --- | --- | --- |
| RNAfold-MFE | 0.454 | 0.368 | 0.252 |
| nolm+phys | 0.475 | 0.388 | 0.315 |
| randlm+phys | 0.461 | 0.379 | 0.275 |
| unet+phys (RiNALMo) | 0.485 | 0.464 | 0.598 |
| b2+phys (RNA-FM) | 0.474 | 0.420 | 0.518 |

*Table 6. Test macro-F1 by family-size tier (training sequences of the family), three seeds. The three tiers together cover all 1,000 Test sequences.*

### 4.7 Partial fine-tuning (supplement)

In a matched small-budget setting (1,500 steps, 163.8K trainable parameters), LoRA on the last four layers improved over the frozen backbone on Test by +0.052 (0.016–0.087; Holm-adjusted p = 0.018), but not significantly on any OOD split (GenA +0.018, GenC +0.001, GenF +0.008). A parameter-matched adapter did not improve over frozen (Test +0.014, −0.006 to 0.030). LoRA minus adapter on Test was +0.038 (−0.001 to 0.088; not significant after correction). The budget is small, so this does not test full fine-tuning.

### 4.8 A second backbone shows the same pattern, with a smaller gain

We repeated the main comparison with a second frozen backbone, RNA-FM (99.5M parameters, 1280-dim embeddings; Chen et al., 2022), embedded the same way and paired with physics features (b2+phys, three seeds). Its Test macro-F1 was 0.488 and its mean OOD macro-F1 was 0.348, against 0.363 and 0.338 without an LM. The gain over no LM was +0.124 on Test (95% CI 0.037–0.190) and +0.010 OOD (−0.013 to 0.027); the ID–OOD interaction was +0.114 (0.027–0.181). By family-size tier the gain was −0.001 (<10), +0.033 (10–49) and +0.203 (≥50; 0.087–0.260), and with equal weight per family it was +0.010 on Test (−0.004 to 0.023). The pattern matches RiNALMo — the gain is concentrated in the large families and absent out of distribution — but is weaker at every tier.

## 5. Discussion

**What the gain consists of.** Our results are consistent with the view that the ID gain of an RNA-LM is the product of two ingredients: pretrained knowledge (it disappears with random weights) and exposure of the decoder to the same family (it grows with the family's training count and shrinks under a cap). This is a hypothesis, not a proven decomposition. The test that would settle it, a random-weight LM under the same per-family cap, has not been run. We also do not claim that the gain is memorization: the interaction with OOD stayed positive under the cap (+0.047), and exposure is confounded with properties of the family itself, which the family fixed-effects model could not separate (p = 0.058).

**Why we do not reproduce the collapse reported for fine-tuned models.** CHANRG reports ID F1 of 0.78 and OOD retention of about 0.31 for a fully fine-tuned RiNALMo-giga U-Net. Our frozen-backbone model reaches ID F1 of 0.55 and retention of 0.70 (0.55–0.99). A plausible explanation is that full fine-tuning lets the model store much more family-specific information, which raises the ID score and lowers retention. We did not test this, and our subset training, small decoder and short schedule differ from the published setup in other ways too. The two results should therefore be read as measurements of different regimes, not as a contradiction.

**Consequences for reporting.** A sequence-weighted ID score is dominated by a few large families: five families account for about half of the Test sequences, and removing them cuts the LM gain from +0.183 to +0.072. We recommend that benchmark papers report, next to the sequence-weighted score, (1) F1 with equal weight per family, (2) F1 per family-size tier, and (3) the number of training sequences of each test family. With these, an ID score can be read as "how well the model does on families it has seen often" and not as a single measure of ability.

## 6. Limitations

- **Scope.** Two backbones (RiNALMo-giga in every analysis, RNA-FM in the main comparison only), one dataset (CHANRG), frozen embeddings, and a small decoder. Results may differ for other models and for full fine-tuning.
- **Subsets and seeds.** Training uses 12,000 sequences and evaluation uses 800–1,000 sequences per split. Main comparisons use three seeds, the cap experiment two, and the scale curve two or three. Seeds estimate training noise only.
- **Few units on GenC and GenF.** The OOD splits GenC and GenF have 11 and 10 resampling units, and the ≥50 tier has 11 families, so intervals there are wide or fragile.
- **Exposure is confounded with family.** Large families differ from small ones in more than training count. The cap reduced the gain by about 60%, not entirely, and a within-family estimate was not significant.
- **Exploratory analyses.** The exposure, cap, weighting and scale analyses were designed after the main results and are outside the Holm correction.
- **No direct comparison with published numbers.** We compare subsets with published full-split values. The RNAfold anchor supports comparability of the data pipeline, not of the protocol.
- **No claim about the mechanism.** We did not show that the gain is memorization, and we did not run the random-weight LM under the cap.

## 7. Data and code availability

The pipeline notebook (CHANRG_Paper_Track_v7_DriveBacked.ipynb), the analysis JSON files, the final report tables and the per-stage run logs are available at github.com/nimaone/rnalm-family-exposure; a Zenodo archive with a DOI will follow \[DOI to add\]. CHANRG and RiNALMo are third-party resources; we used the multimolecule/chanrg Hugging Face repository, whose dataset card lists AGPL-3.0-or-later while the CHANRG paper is CC BY-NC-SA 4.0, so no dataset content is redistributed here. RiNALMo code is Apache-2.0 and its weights are CC BY 4.0 (Zenodo record 15043668); the RNA-FM checkpoint is AGPL-3.0-or-later. Both backbones were loaded through MultiMolecule (Chen and Zhu, 2024).

**Competing interests.** The author declares no competing interests.

**Use of AI tools.** In preparing this manuscript, the author used an AI assistant (Claude, Anthropic) for developing the analysis code and for drafting and editing the text. All AI-assisted content was reviewed and edited by the author, who approved the final manuscript and takes full responsibility for its content and for the integrity of the reported results.

## References

- Penić, R. J., Vlašić, T., Huber, R. G., Wan, Y., Šikić, M. (2025). RiNALMo: general-purpose RNA language models can generalize well on structure prediction tasks. *Nature Communications* 16, 5671. [doi:10.1038/s41467-025-60872-5](https://doi.org/10.1038/s41467-025-60872-5).
- Chen, Z., Deng, Z., Deng, P., Liao, Y., Su, X., Ye, P., Liu, X. (2026). Fair splits flip the leaderboard: CHANRG reveals limited generalization in RNA secondary-structure prediction. [arXiv:2603.22330](https://arxiv.org/abs/2603.22330).
- Szikszai, M., Wise, M., Datta, A., Ward, M., Mathews, D. H. (2022). Deep learning models for RNA secondary structure prediction (probably) do not generalize across families. *Bioinformatics* 38(16), 3892–3899. [doi:10.1093/bioinformatics/btac415](https://doi.org/10.1093/bioinformatics/btac415).
- Zablocki, L. I., Bugnon, L. A., Gerard, M., Di Persia, L., Stegmayer, G., Milone, D. H. (2024; v2 2025). Comprehensive benchmarking of large language models for RNA secondary structure prediction. [arXiv:2410.16212](https://arxiv.org/abs/2410.16212).
- Galvez, G., Vicens, Q. (2026). Shifu: an integrated framework for deep learning of RNA secondary structure. bioRxiv. [doi:10.64898/2026.07.15.738762](https://www.biorxiv.org/content/10.64898/2026.07.15.738762v1).
- Chen, J., Hu, Z., Sun, S., Tan, Q., Wang, Y., Yu, Q., Zong, L., Hong, L., Xiao, J., King, I., et al. (2022). Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions. [arXiv:2204.00300](https://arxiv.org/abs/2204.00300).
- Chen, Z., Zhu, S. Y. (2024). MultiMolecule. Zenodo. [doi:10.5281/zenodo.12638419](https://doi.org/10.5281/zenodo.12638419).
