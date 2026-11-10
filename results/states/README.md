# Traffic states (Fig. 1(d), Fig. 6)

| File | Content |
|---|---|
| `state_comparisons_test.csv` | MAE reduction of the target-only module and of DeCo-S against the single DecoupleST forecasters, by state: speed and occupancy at level S2 and speed at the two looser levels. Section V-F quotes the speed reductions at S2, and Fig. 1(d) draws them for onset, congestion and recovery. `share_of_valid_targets` is the state's share of that variable's valid targets. |
| `per_seed.csv` | MAE per seed, level, state, variable and method (single forecasters, target-only module, DeCo-S), for the same rows. Fig. 6 shows the speed MAE reduction of DeCo-S against the target-only module at the three levels, computed from the three-seed means of these values. |
| `thresholds.json` | Congestion thresholds per dataset and level (speed at most, occupancy at least). |

- **States:** a sensor is congested at a step when speed is at most and occupancy at least the thresholds. A target is in free flow, at onset, in congestion or in recovery when neither, only the target step, both, or only the last observed step is congested. Targets with an invalid occupancy or speed at either step have no state.
- **Levels:** quantiles of the training targets pooled over sensors: original (speed q0.35, occupancy q0.65; PeMS04 keeps its earlier constants 63.9 mph and 0.0598), S1 (q0.20 / q0.80), S2 (q0.05 / q0.95). `python run.py thresholds --dataset <name>` recomputes them.
- **Methods:** one forecaster per variable for all three (FINAL deployment): the first-phase DecoupleST SISO forecasters, the same with the target-only module, and DeCo-S.
- The per-seed values were recomputed with `python run.py states` from the saved test forecasts. PeMS04 and PeMS08 were evaluated on float32 labels in the reported runs and on the float64 arrays of `data/make_pems0408.py` here; the state of every target is the same, and the reductions agree to within 0.001 percentage points.
