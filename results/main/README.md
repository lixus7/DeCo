# Main comparison (Tables III and IV)

| File | Content |
|---|---|
| `main_table_test.csv` | Test MAE and RMSE (mean, SD) of the 14 rows of Table III on every dataset and variable (168 rows). `mae_rank` and `rmse_rank` mark the bold (`best`) and underlined (`second`) cells: lowest and second lowest of the 14 rows at the printed precision, ties sharing the mark. |
| `per_seed.csv` | MAE and RMSE per seed for these rows, with and without zero targets. DeCo has test values only; the other rows also have validation values. |
| `further_baselines_test.csv` | Test metrics of every trained form of the 17 further baselines on PeMS04 and PeMS08 (276 rows). `lowest_mae_of_model` marks the form shown in Table IV, the lowest test MAE among a model's forms. |
| `further_baselines_per_seed.csv` | MAE and RMSE per seed for those forms, for validation and test, with and without zero targets. |

- **Variants:** SISO (1→1) predicts a variable from its own history; MISO (3→1) from the histories of all three variables, one model per variable; MIMO (3→3) predicts all three with one model. DeCo and DeCo (RAGC) are labelled MIMO, as in the paper.
- **DecoupleST SISO** is the single forecaster per variable trained on the full training split; **DecoupleST MISO** is its early-fusion form.
- The DeCo (RAGC) rows also appear in `../ragc_extension/` (coordination module, K12 deployment).
