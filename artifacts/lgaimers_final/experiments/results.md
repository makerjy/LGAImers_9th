# Experiment results

## Evaluation contract

- Source: `final/v14_oof.parquet` joined to official `train.csv` by unique `row_id`.
- Folds represented: 2022, 2023, 2024 outer temporal OOF rows already stored in the artifact.
- Primary selection fold: 2024.
- Stress fold: 2023.
- Reference rate for BSS proxy: validation target mean, because the real test target mean is unavailable.
- Paired confidence intervals: pitcher-cluster bootstrap, 2,000 resamples.

## Overall measured candidates

| candidate_id | brier | bss_proxy | auc | prediction_mean | target_mean | ci_low | ci_high | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V13_baseline | 0.246641 | 1334.487885 | 0.559065 | 0.504160 | 0.504855 | 0.000000 | 0.000000 | MEASURED |
| B_mh_profile | 0.247598 | 951.504768 | 0.557734 | 0.513439 | 0.504855 | 0.000629 | 0.001322 | MEASURED |
| C_tabm_direction | 0.247829 | 859.196407 | 0.552937 | 0.498082 | 0.504855 | 0.000858 | 0.001545 | MEASURED |
| D_v14_uncalibrated | 0.246279 | 1479.117567 | 0.563347 | 0.506064 | 0.504855 | -0.000458 | -0.000277 | MEASURED |
| E_v14_month_calibrated | 0.246189 | 1515.043581 | 0.563890 | 0.504855 | 0.504855 | -0.000557 | -0.000350 | MEASURED |

`ci_low` and `ci_high` are the 95% interval for candidate Brier minus the V13 baseline.
Negative values favor the candidate.

## R/F subgroup

| candidate_id | segment | brier | bss_proxy | auc |
| --- | --- | --- | --- | --- |
| V13_baseline | R | 0.248388 | 644.300523 | 0.545829 |
| V13_baseline | F | 0.233247 | 5700.135619 | 0.631013 |
| B_mh_profile | R | 0.248060 | 775.236256 | 0.549929 |
| B_mh_profile | F | 0.244052 | 1331.508443 | 0.596325 |
| C_tabm_direction | R | 0.248035 | 785.556808 | 0.549845 |
| C_tabm_direction | F | 0.246250 | 443.114744 | 0.591599 |
| D_v14_uncalibrated | R | 0.247979 | 807.782737 | 0.550818 |
| D_v14_uncalibrated | F | 0.233247 | 5700.135619 | 0.631013 |
| E_v14_month_calibrated | R | 0.247921 | 831.103535 | 0.551247 |
| E_v14_month_calibrated | F | 0.232915 | 5834.084117 | 0.633248 |

## Feature ablation

The stored OOF artifact contains composite predictions. It contains no matched
profile-only/no-season-to-date pair, so an isolated feature ablation is not claimed.
The measured sequence is V13 baseline → MH profile composite → V14 direction blend →
month calibration.

| ablation step | comparable prediction file | status | conclusion |
|---|---|---|---|
| V13 baseline | `p_v13_centered_expert_best` | MEASURED | reference anchor |
| MH profile composite | `p_mh_raw_blend` | MEASURED | profile and season-to-date are coupled |
| profile-only vs no-season-to-date | unavailable | UNAVAILABLE | no same-fold pair in stored OOF |
| season-to-date isolated lift | unavailable | UNAVAILABLE | reconstruction passed; isolated lift not claimed |
| V14 direction blend | `p_v14_uncalibrated` | MEASURED | improves pooled Brier vs V13 |
| month calibration | `p_v14_month_calibrated` | MEASURED | pooled improvement; nested caveat |

## Year weighting

The current measured MH/V14 artifact uses the four-year weights `1.00, 0.75, 0.50,
0.25`. Equal, two-year, three-year, five-year, and all-period alternatives do not have
same-fold prediction files in the current workspace and are marked unavailable rather
than recreated from a different protocol.

| year weighting | same-fold result | status | note |
|---|---|---|---|
| equal | unavailable | UNAVAILABLE | not stored |
| recent 2 years | unavailable | UNAVAILABLE | not stored |
| recent 3 years | unavailable | UNAVAILABLE | not stored |
| recent 4 years `1.00/0.75/0.50/0.25` | `B_mh_profile`, `D/E_v14` | MEASURED | current composite artifact |
| recent 5 years decay | unavailable | UNAVAILABLE | not stored |
| all-period decay | unavailable | UNAVAILABLE | not stored |

## Calibration

The stored uncalibrated and month-calibrated predictions are compared above. The month
calibration improves pooled OOF Brier, but the nested next-season diagnostic in
`final/v14_results.json` is worse than V13. It is not treated as official generalization.

| calibration candidate | pooled Brier | 2023 Brier | 2024 Brier | status |
|---|---:|---:|---:|---|
| none / V14 uncalibrated | 0.246279 | 0.248180 | 0.247572 | MEASURED |
| game_type × month affine | 0.246189 | 0.248079 | 0.247537 | MEASURED, nested caveat |
| Platt / isotonic | unavailable | unavailable | unavailable | UNAVAILABLE in current artifact |

## HCCN paired comparison

| candidate_id | status | notes |
| --- | --- | --- |
| A_v5_hccn | BLOCKED | v5/HCCN inference artifact exists, but matched 2022-2024 OOF predictions are absent. |
| F_hccn_residual_incremental | BLOCKED | HCCN source/checkpoints are preserved, but the comparable raw OOF table is not. |
| G_trackman_student | BLOCKED | Current final source and feature schema do not use Trackman directly. |

Raw HCCN/v5 OOF predictions are not present. No HCCN incremental gain is reported.

## Automatic selection

```json
{
  "selected_candidate": "E_v14_month_calibrated",
  "selection_reason": "Lowest 2024 Brier among measured candidates within the 2023/2024 guardrails.",
  "primary_fold": "2024",
  "stress_fold": "2023",
  "reference_anchor": "p_v13_centered_expert_best",
  "hccn_incremental_candidate_selected": false,
  "hccn_inherited_in_v14_runtime": true,
  "hccn_decision": "No isolated HCCN residual lift is claimed because comparable raw HCCN OOF is absent; the stored v14 inference artifact still executes the inherited Track A -> HCCN/v5 -> V13 path.",
  "season_to_date_included": true,
  "season_to_date_audit": "feature reconstruction report; current final OOF is composite profile+season-to-date",
  "blocked_candidates": [
    "A_v5_hccn",
    "F_hccn_residual_incremental",
    "G_trackman_student"
  ],
  "nested_caveat": "V14 pooled calibration improves local OOF but stored nested diagnostic is weaker."
}
```
