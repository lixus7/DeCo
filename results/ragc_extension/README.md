# DeCo with RAGC forecasters (Table VI)

The coordination module trained on RAGC (Wu et al., ICDE 2026) forecasters instead of DecoupleST: twelve RAGC fold forecasters per variable and the module, without the second training phase and the early-fusion forecaster. We decided to run this extension after inspecting the main test results and a test of the module applied to RAGC's forecasts without refitting; the training recipe and reporting rules were fixed before the runs. All 24 evaluations (4 datasets, 3 seeds, coordination and target-only modules) are reported.

| File | Content |
|---|---|
| `ragc_table_test.csv` | Table VI: test MAE (three-seed mean) of RAGC, RAGC + coordination (one forecaster per variable), the twelve-fold RAGC average, + target-only and + coordination (twelve-fold; DeCo (RAGC) in Table III), and DeCo; `deco_vs_twelve_fold_average_pct` is DeCo's MAE reduction against the twelve-fold RAGC average. |
| `summary_test.csv` | Test metrics for every module, deployment, dataset and variable (72 rows). |
| `comparisons_test.csv` | Paired test MAE comparisons for the K12 and FINAL rows (240 rows). |
| `per_seed.csv` | MAE, RMSE and selected epoch per seed, for validation and test, with and without zero targets (576 rows). |
| `fold_mean_comparisons_test.csv` | Paired test MAE comparisons with RAGC's twelve-fold average without the module (48 rows). |
| `fold_mean_per_seed_test.csv` | Test MAE and RMSE of that average per seed, with and without zero targets (72 rows). |

- **Module:** `coordination` reads all three variables at the sensor; `target_only` reads copies of the target variable.
- **Deployments:** `K12` corrects each of the twelve RAGC fold forecasters per variable and averages the corrected forecasts; `FINAL` corrects the single RAGC fit of Table III; `TRANSFER` applies the K12-selected checkpoint to that single fit.
- **References in the comparison files:** `DeCo-S` and `K12 + coordination on DecoupleST (first phase)` (twelve first-phase DecoupleST fold forecasters with the module with sensor embedding) are DecoupleST-based systems; `best of 11 baselines` is the baseline with the lowest test MAE in that cell.
- **Intervals:** 95% day-block intervals with 10,000 resamples, per cell, omitting training randomness, not corrected for multiple comparisons.
- `TRANSFER` values come from run records, with no day-block analysis, so `comparisons_test.csv` does not include them.
- RAGC's twelve-fold average is an unranked reference; no ranked baseline in Table III is given a fold average. The fold forecasters and the single fit are trained on different data, so the comparison of the two is cross-fitted averaging against a single fit, not the averaging operation alone.
