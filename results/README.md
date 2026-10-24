# Results artifacts

Everything here was produced by `code/CHANRG_Paper_Track_v7_DriveBacked.ipynb` on Google Colab
(T4). Bootstrap settings throughout: 4,000 resamples over resampling units × seeds.

| File | Producing stage | Contents |
| --- | --- | --- |
| `analysis_v5.json` | P7/P9 | Main confirmatory statistics: per-arm F1 gaps (macro/micro), 36 Holm-corrected tests, LM / pretrain / random-feature interactions, split-unit info (Test 224 families, GenA 99 super-families, GenC 11 clans, GenF 10 clans) |
| `analysis_v5_cap.json` | P10d | Per-family cap=10 experiment: gains under full / size-matched random subset / cap, overall contrasts |
| `analysis_v5_exposure_fixed.json` | P10b | LM gain by family-exposure tier and the family fixed-effects regression (within-family slope) |
| `analysis_v5_exposure_scale.json` | P10c | Exposure tiers computed across the training-scale curve (1,500/3,000/6,000/12,000) |
| `analysis_v5_scale_exposure.json` | P10 | Scale-curve F1 per arm and size, plus exposure view |
| `report_tables_v5.md` | P12a | Final tables: F1 per system (sequence-weighted and family-equal), F1 by family-size tier, gains vs `nolm+phys` with CIs, including `b2+phys` (RNA-FM) and LoRA. Verbatim run output (Persian labels) |
| `logs/` | all | Raw per-stage Colab logs; the definitive record of what ran, when, and with how many seeds |

## Reading the logs

Log names follow the stage names (`p0_env`, `p1_data`, `p1b_leakage`, `p2_embed`,
`p2r_embed_random`, `p2b_embed_backbone2`, `p3_phys`, `p4_reference`, `p5_main`,
`p5r_randlm`, `p5s_scale`, `p5c_cap`, `p5x_*`, `p6_lora`, `p6b_adapter`,
`p6l_lora_full`, `p7`–`p10*`, `p12a_tables`, `p12b_dose_factorial`).
Gates (`GATE: PASS/FAIL`) assert pre/post-conditions (leakage checks, LoRA
gradient flow, cache equivalence); a stage prints its saved artifact paths at
the end.

Two known incidents, kept deliberately:

- `logs/p5x_bb2_full-randlm_cap-seed3_b99834.log` — the `randlm_cap` / `seed3`
  / caps-3-and-50 jobs aborted because random-LM embeddings (`emb_rand`) were
  missing in that environment. Re-running P2r in the same session before P5x
  fixes it.
- `logs/p6l_lora_full.log` vs the LoRA row in `report_tables_v5.md` — the two
  disagree on both the numbers (Test 0.568 vs 0.548; OOD 0.388 vs 0.354) and
  the schedule label (24,000 updates, seed 0, vs "12000×3"). Unresolved; the
  manuscript does not use these numbers.
