# Cost (Table VIII)

| File | Content |
|---|---|
| `inference_test.csv` | Parameters, median latency (ms), forecast windows per second and peak PyTorch-allocated GPU memory, at batch sizes 1 and 16, on PeMS04 and PeMS08. |
| `training_hours.csv` | H100 hours per seed (mean over the three seeds) on the four datasets. |
| `training_hours_per_seed.csv` | The same per seed. |

- **Configurations:** `DecoupleST (x3)` is the three single forecasters (one per variable); `DeCo-S` is 36 fold and 3 final first-phase forecasters and its module; `DeCo` is 36 fold, 3 final and 3 early-fusion forecasters, their second training phase and the module.
- **Latency:** one NVIDIA H100, float32, TF32 disabled, PyTorch 2.4.1, CUDA 12.1; ten warm-up calls, then the median of 100 synchronized calls from GPU-resident inputs to the final forecast, on reference checkpoints with the same forward paths. Parameter counts exclude the unused sensor-embedding table of DeCo's module.
- **Training hours:** sums of the epoch wall times of the accepted fits, including validation and data access; every listed fit ran on an H100. Three resumed PeMS07 module runs of DeCo did not record the part of an epoch run before a pause; that time is estimated from the job records (`includes_estimated_time`).
