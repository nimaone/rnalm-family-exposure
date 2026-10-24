# Family exposure explains most of the in-distribution advantage of RNA language models on CHANRG

Companion code and results for:

> **How much of an RNA language model's in-distribution advantage is family exposure? A controlled decomposition on CHANRG**
> Nima Baselizadeh — Bu-Ali Sina University, Hamedan, Iran
> Correspondence: nimabaselizadeh@gmail.com

Full manuscript (Markdown / PDF / DOCX) is in [`manuscript/`](manuscript/). Preprint and DOI will be added on deposition.

## What the paper does

Pretrained RNA language models (RNA-LMs) are reported to generalize poorly to RNA families absent from training. On the CHANRG benchmark (50–200 nt) we decompose the in-distribution (ID) advantage of a frozen RNA-LM paired with a fixed lightweight U-Net pair-map decoder by changing **one factor at a time**:

| Intervention | Question |
| --- | --- |
| no-LM baseline (`nolm+phys`) | What does the physics feature set alone achieve? |
| random-weight LM (`randlm+phys`) | Is the gain pretrained knowledge or just a high-dimensional input? |
| second frozen backbone, RNA-FM via MultiMolecule (`b2+phys`) | Does the pattern generalize beyond one LM? |
| per-family training caps + size-matched random subsets | How much of the ID gain is exposure to the same family? |
| nested training-scale curve (1.5k–12k) | Does the ID gain grow with data while the OOD gain does not? |
| per-family (equal-weight) reporting | How much of the headline number comes from a few large families? |

Headline results (macro-F1, mean of 3 seeds): frozen RiNALMo-giga adds **+0.18** ID (95% CI 0.09–0.24) but only **+0.04** OOD (0.01–0.07); a random-weight LM gives **no** benefit; capping each family at 10 training sequences removes ~60% of the ID gain; weighting families equally shrinks the ID gain to +0.04. Our RNAfold pipeline reproduces published length-matched baselines to within 0.014 F1. See the manuscript for full statistics (bootstrap over resampling units × seeds, Holm-corrected confirmatory family).

## Repository contents

| Path | Contents |
| --- | --- |
| `manuscript/` | The paper (`.md`, `.pdf`, `.docx`) |
| `code/CHANRG_Paper_Track_v7_DriveBacked.ipynb` | The complete, self-contained experiment pipeline (Colab, Drive-backed resume). This single notebook produced every number in the paper |
| `results/analysis_v5.json` | Main statistics: gaps per arm, 36 Holm-corrected confirmatory tests (macro/micro), interactions, resampling-unit info |
| `results/analysis_v5_cap.json` | Per-family cap experiment (cap=10 vs size-matched random subset vs full training set) |
| `results/analysis_v5_exposure_fixed.json` | LM gain by family exposure tier + family fixed-effects regression |
| `results/analysis_v5_exposure_scale.json` | Exposure analysis across the training-scale curve |
| `results/analysis_v5_scale_exposure.json` | Scale-curve F1 per arm/size |
| `results/report_tables_v5.md` | Final report tables (sequence-weighted, family-equal, family-size tiers, gains vs `nolm+phys`, incl. `b2+phys` and LoRA) |
| `results/logs/` | Raw per-stage run logs from Colab (all stages, latest run Oct 15, 2026) |

## Pipeline

The notebook is organized in stages. Each stage caches its artifacts to Google Drive and **skips itself** if already complete with the same configuration fingerprint, so runs survive Colab disconnects.

| Stage | What it does |
| --- | --- |
| P0 | Environment, shared library (`plib`), gates |
| P1 / P1b | CHANRG data (gated HF download), subset sampling, leakage gates, resampling units |
| P2 / P2r / P2b | Embeddings: RiNALMo-giga (frozen, fp16), random-weight RiNALMo, second backbone RNA-FM (`multimolecule/rnafm` by default) |
| P3 | ViennaRNA 2.7.2 physics feature cache |
| P4 | RNAfold-MFE / base-pair-probability reference + anchor check against published length-matched values |
| P5 / P5r / P5s / P5c | Main arms (`unet+phys`, `unet`, `nolm+phys`, `randlm+phys`), scale curve, cap experiment |
| P6 / P6b / P6L | LoRA (r=8, layers 29–32), parameter-matched adapter control, full-scale LoRA |
| P5x | Extra jobs: second-backbone full run, random-LM under cap, third seed, caps 3/50 |
| P7–P10 | Statistics, report gates, Holm family, exposure/scale/retention analyses |
| P12a / P12b | Final report tables; dose–response over caps and 2×2 (pretraining × exposure) test |

## How to run

1. Open `code/CHANRG_Paper_Track_v7_DriveBacked.ipynb` in **Google Colab** (T4 GPU is enough; full profile ≈ several hours across sessions).
2. Set your Hugging Face token with access to the gated `multimolecule/chanrg` dataset (Colab Secrets, `HF_TOKEN`, or `/content/hf_token.txt`).
3. Run the **settings cell** first: `PROFILE="lit"` reproduces the paper (12,000 training sequences, 3 epochs, 3 seeds, U-Net width 32); `PROFILE="quick"` is a smoke test. `DRIVE_RUN` names the Google Drive run folder.
4. Run stages in order P0 → P1 → P1b → P2 → P3 → P4 → P5 → P2r → P5r → P5s → P7–P10; P2b/P5x/P6*/P12* as needed.
5. Key environment variables (see the settings cell for defaults): `RNA_TRAIN_N`, `RNA_EPOCHS`, `RNA_SEEDS`, `RNA_SCALE_SIZES`, `RNA_SCALE_SEEDS`, `RNA_RAND_SEED`, `RNA_CAPS`, `RNA_XP_JOBS`, and `RNA_BB2` for the second backbone (default `multimolecule/rnafm`; `multimolecule/ernierna` also supported — note its AGPL-3.0 license).

Note: the notebook's first markdown cell still carries the historical title "CHANRG Paper Track v5"; the file is the final v7 Drive-backed version that produced the results archived here.

## Status of the latest runs (Oct 15, 2026)

- `b2+phys` (RNA-FM second backbone, full training, 3 seeds): **complete** — see `results/report_tables_v5.md` (Test +0.124 over no-LM, OOD +0.010, ID–OOD interaction +0.114, tier ≥50 gain +0.203).
- **P5x partially failed**: jobs `randlm_cap` and `seed3`/`RNA_CAPS="3,50"` aborted with `FileNotFoundError: /content/rna_paper_v4/emb_rand/train.npy` (random-LM embeddings were not present in that environment — see `results/logs/p5x_bb2_full-randlm_cap-seed3_b99834.log`). To reproduce, re-run P2r in the same session before P5x. Consequently the P12b dose–response currently covers **cap=10 only**, and the random-weight-LM-under-cap cell of the 2×2 test is still open.
- **P6L (full-scale LoRA) has an unresolved inconsistency**: the run log (`results/logs/p6l_lora_full.log`; seed 0, 2 epochs = 24,000 updates) reports Test=0.568 / OOD=0.388, while the P12a table for the same system reports Test=0.548 / OOD=0.354 and labels it "12000×3". Until clarified, the full-scale LoRA numbers are **not** used in the manuscript.

## Data and model availability

- **CHANRG** is third-party and gated: `multimolecule/chanrg` on Hugging Face (dataset listed AGPL-3.0-or-later; the CHANRG paper is CC BY-NC-SA 4.0). It is **not** redistributed here.
- **RiNALMo-giga** and **RNA-FM** are loaded through [MultiMolecule](https://github.com/multimolecule/multimolecule) at runtime; no weights are redistributed. RiNALMo code is Apache-2.0, weights CC BY 4.0 (Zenodo 15043668). Check the MultiMolecule model cards for the RNA-FM checkpoint license.
- This repository distributes only **code, run logs and aggregate analysis JSONs** (no CHANRG sequences, no per-sequence predictions, no model weights).
- A Zenodo archive with DOI will be added on deposition.

## Citation

```bibtex
@software{baselizadeh2026chanrg_exposure,
  author  = {Baselizadeh, Nima},
  title   = {How much of an RNA language model's in-distribution advantage is family exposure? A controlled decomposition on CHANRG},
  year    = {2026},
  url     = {https://github.com/nimaone/rnalm-family-exposure}
}
```

(Replace with the versioned Zenodo DOI once archived. See also `CITATION.cff`.)

## License

Code in this repository: **MIT** (see `LICENSE`).
Everything derived from CHANRG (splits definitions, published baseline numbers) remains under its upstream licenses (AGPL-3.0-or-later dataset / CC BY-NC-SA 4.0 paper); the manuscript in `manuscript/` is the author's copy.
