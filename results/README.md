# Results

Test results behind the tables of the paper, with the per-seed values they are computed from. Every printed cell of Tables III–VIII was recomputed from these files and matches the paper. DeCo's configuration and the analyses were chosen during development with access to the test results of all four benchmarks.

| Folder | Paper | Content |
|---|---|---|
| `main/` | Tables III and IV | Main comparison (eleven baselines, DecoupleST, DeCo, DeCo (RAGC)) and the 17 further baselines |
| `ablation/` | Table V | Components of DeCo (a) and inputs of the coordination module (b) |
| `ragc_extension/` | Table VI | DeCo with RAGC forecasters, with the complete results of that extension |
| `states/` | Table VII, Fig. 5 | MAE by traffic state at three congestion levels |
| `cost/` | Table VIII | Parameters, latency and training hours |

## Conventions

- **Metrics:** MAE and RMSE over valid targets, keeping valid zeros (`all valid`, the primary metric); `valid, excluding zero` drops zero targets.
- **Units:** flow in vehicles per five minutes, occupancy as a fraction (the paper shows percent), speed in mph.
- **Seeds:** 2024, 2025 and 2026. Means and SDs are the mean and sample SD of the three seed-level scores.
- **Changes and reductions:** in percent of the reference MAE; a positive MAE reduction favours the candidate.
- **Names:** DeCo is the full system; DeCo-S is one forecaster per variable with the coordination module (first-phase forecasters, module with sensor embedding); the target-only module reads copies of the target variable only; DeCo (RAGC) replaces DeCo's forecasters with twelve RAGC fold forecasters per variable, without the second training phase and the early-fusion forecaster.
