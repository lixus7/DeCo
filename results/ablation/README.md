# Ablation (Fig. 3)

| File | Content |
|---|---|
| `ablation_test.csv` | Test MAE (three-seed mean and SD) of the variants of Fig. 3(a) and (b) and their change against the first row of their block, which the bars show; the rows `average of the 12 columns` give the averages over datasets and variables quoted in Section V-C. |
| `per_seed.csv` | MAE and RMSE per seed of these rows, with and without zero targets (test; validation where it was scored), and of the two first-phase twelve-fold rows quoted in Section V-C. |
| `coordination_intervals_test.csv` | DeCo against DeCo without the coordination module: MAE reduction with its 95% day-block interval and the number of seeds with a lower MAE. |

**(a) Components of DeCo.** Each row compares a whole system with DeCo: `w/o early-fusion branch` is the fold average of the corrected forecasts; `w/o coordination` blends the uncorrected fold average with the early-fusion forecasts; `w/o both` is the fold average of the second-phase forecasters; `w/o both, w/o second training phase` is the fold average of the first-phase forecasters; `single forecaster per variable` is DecoupleST SISO.

**(b) Inputs of the coordination module.** With one forecaster per variable (DeCo-S): the two input rows set the other variables' histories or forecasts to zero in the module's standardized coordinates and retrain the module (arms `coord_no_hist`, `coord_no_fc`; PeMS04 and PeMS08); `w/o coordination module (single forecaster)` is DecoupleST SISO; `input fusion` is DecoupleST MISO.
