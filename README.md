# DeCo

Code, data and results for *DeCo: Decoupled Traffic State Modeling with Cross-Variable Coordination* (submitted to ICDE 2027).

## Contributions

1. **DeCo, a decoupled forecaster with cross-variable coordination.** One forecaster per traffic variable (flow, occupancy, speed) is trained on that variable alone. A small coordination module then reads the recent histories and the forecasts of all three variables at a sensor and corrects each forecast. The module is trained on out-of-fold forecasts while the forecasters stay fixed. At inference, the module corrects the forecasts of the twelve fold forecasters per variable, the corrected forecasts are averaged, and the average is blended with an early-fusion forecaster at a fixed weight of 0.2 (Algorithm 1 of the paper). With the same forecasters, removing the module raises the error on every dataset and variable, and DeCo has the lowest MAE on every dataset and variable against eleven baselines, by up to 4.2% over the best one (Tables III and V).
2. **Three-variable PeMS03 and PeMS07.** The PeMS03 and PeMS07 benchmarks carry flow only. We align occupancy and speed from the Caltrans PeMS station records to the benchmark flow series, keeping their sensor sets and flow records. Where PeMS reports flow, it matches the benchmark flow in 99.956% of the PeMS03 entries and in every PeMS07 entry. The aligned arrays, their validity masks, station identifiers and source checksums are in the [`data-v1` release](https://github.com/lixus7/DeCo/releases/tag/data-v1); `data/README.md` describes them.
3. **A controlled comparison of input and output forms.** STAEformer, HimNet and D²STGNN are trained in the single-variable (SISO, 1→1), auxiliary-input (MISO, 3→1) and joint-output (MIMO, 3→3) forms on all four benchmarks under one protocol. Auxiliary inputs lower the MAE in most settings, by up to 2.9%, whereas joint prediction raises it in most settings, by up to 8.2% (Table III and Fig. 1(c) of the paper). The comparison separates cross-variable coordination from target-only correction and forecast averaging (Table V).

## Layout

| Path | Content |
|---|---|
| `run.py` | Command-line entry for every stage |
| `deco/decouplest.py` | DecoupleST forecaster |
| `deco/train.py` | First training phase of every forecaster and baseline (at most 200 epochs, patience 30) |
| `deco/second_phase.py` | Second training phase: ten epochs with embedding swap, weights averaged |
| `deco/fold_inputs.py`, `deco/prepare.py` | Twelve-fold partitions and fold-local causal inputs |
| `deco/banks.py` | Out-of-fold, fold and final forecasts of one phase in one bank |
| `deco/module.py` | Coordination module, target-only module, the variant without sensor embedding and the input-zeroing views |
| `deco/module_train.py`, `deco/module_data.py`, `deco/selection.py` | Module training, checkpoint selection on validation, and the fold-average (K12) and single-forecaster (FINAL) deployments |
| `deco/blend.py` | DeCo's output: fold average of the corrected forecasts blended with the early-fusion forecasts |
| `deco/evaluation.py`, `deco/states.py` | Metrics with validity masks, traffic states, congestion thresholds and day-block comparisons |
| `deco/models.py`, `deco/external_models.py` | DecoupleST MISO, RAGC, the module arms and the baseline wrappers |
| `configs/` | Dataset settings, congestion thresholds and baseline source commits |
| `data/` | Data layout, checksums and provenance; the arrays are in the `data-v1` release |
| `results/` | Results behind Tables III–VIII of the paper, with per-seed values (`results/README.md`) |
| `tests/smoke_cpu.py` | End-to-end check of every stage on synthetic data |

## Setup

Python 3.11 with the packages in `requirements.txt`. See `third_party/README.md` for the baselines, whose code is not redistributed here.

The aligned PeMS03 and PeMS07 arrays are in the [`data-v1` release](https://github.com/lixus7/DeCo/releases/tag/data-v1). `data/README.md` describes how to download them, how to build PeMS04 and PeMS08 from their public release, and how the arrays were made.

To check that the code runs without the data (a few minutes on CPU):

```bash
python tests/smoke_cpu.py
```

## Reproducing the DeCo row of Table III

For one dataset and seed, `pipeline` runs every step in sequence (skipping completed ones) and writes each task to `results/<task id>/`:

```bash
python run.py prepare --dataset pems08
python run.py pipeline --dataset pems08 --seed 2024
```

The DeCo forecast and its test metrics are in `results/deco-pems08-2024/` (`test_prediction.npy`, `COMPLETE.json`). The paper reports the mean and standard deviation over seeds 2024, 2025 and 2026. The same run also trains DeCo-S and the target-only module (`results/eval-a12-pems08-bank2024-s2024-coord/` and `...-target_only/`, FINAL deployment); `--ablation` adds the two input-zeroing arms of Table V(b).

On a cluster, schedule the steps yourself. Per dataset and seed (`--variable` is `flow`, `occupancy` or `speed`, `--fold` 0–11):

```bash
# 1. First phase: 12 fold forecasters and one final forecaster per variable, and one early-fusion (MISO) forecaster per variable
python run.py forecaster --dataset pems08 --seed 2024 --variable flow --fold 0
python run.py forecaster --dataset pems08 --seed 2024 --variable flow
python run.py forecaster --dataset pems08 --seed 2024 --variable flow --architecture decouplest_miso
# 2. Second training phase of each of these 42 forecasters
python run.py continue --dataset pems08 --seed 2024 --variable flow --fold 0
python run.py continue --dataset pems08 --seed 2024 --variable flow
python run.py continue --dataset pems08 --seed 2024 --variable flow --architecture decouplest_miso
# 3. Bank of the second-phase forecasts, DeCo's module, its fold-average deployment, and the blend
python run.py bank --dataset pems08 --seed 2024 --phase 2
python run.py module --dataset pems08 --seed 2024 --phase 2 --arm coord_nonode
python run.py evaluate --dataset pems08 --seed 2024 --phase 2 --arm coord_nonode
python run.py blend --dataset pems08 --seed 2024
```

Variants and analyses:

```bash
# DeCo-S and the target-only module (first-phase bank, FINAL deployment)
python run.py bank --dataset pems08 --seed 2024
python run.py module --dataset pems08 --seed 2024 --arm coord
python run.py evaluate --dataset pems08 --seed 2024 --arm coord
python run.py module --dataset pems08 --seed 2024 --arm target_only
python run.py evaluate --dataset pems08 --seed 2024 --arm target_only
# Input zeroing of Table V(b): the other variables' histories or forecasts set to zero
python run.py module --dataset pems08 --seed 2024 --arm coord_no_hist
python run.py module --dataset pems08 --seed 2024 --arm coord_no_fc
# Traffic states of Table VII (after DeCo-S and the target-only module for all three seeds)
python run.py states --dataset pems08 --levels S2
python run.py thresholds --dataset pems08
# Baselines
python run.py forecaster --dataset pems08 --seed 2024 --variable speed --architecture ragc
python run.py baseline --dataset pems08 --seed 2024 --architecture staeformer --variant miso_speed
```

| Name in the paper | Module arm | Bank (`--phase`) | Deployment |
|---|---|---|---|
| DeCo | `coord_nonode` | 2 (second-phase forecasters) | K12 (fold average), then `blend` |
| DeCo-S | `coord` | 1 (first-phase forecasters) | FINAL (one forecaster per variable) |
| Target-only module | `target_only` | 1 | FINAL |
| w/o other variables' histories / forecasts | `coord_no_hist` / `coord_no_fc` | 1 | FINAL |

The reported DeCo fits ran on NVIDIA H100 GPUs. Across datasets, the three-seed mean training cost ranges from about 14 to 150 H100 hours per seed, most of it in the 36 fold forecasters, which are independent and can run in parallel; per-seed values are in `results/cost/`.

## Implementation details

These details complete Sections IV and V of the paper.

**DecoupleST.** The history-availability vector is (1, 0, 0, 0): only the recent group is filled, and the zero-valued history groups still pass through the network. Each head maps 4×12×64 = 3072 features through 128 hidden units to 12 steps, and the direct map takes the 48 history entries to 12 steps. The adjacency is A = softmax(ReLU(E1 E2ᵀ)) with the diagonal excluded, E1, E2 ∈ R^{N×16}. In the 504 trained first-phase forecasters that we scanned, E1 and E2 are numerically zero, so A is uniform over the other sensors; the weights after the second phase were not scanned.

**Coordination module.** Each variable's encoder reads 60 entries: 12 history steps, 36 zeros for the unused daily and weekly slots, and 12 forecast steps. φ_q has 16 hidden units; the activations are GELU; the LayerNorm affine parameters are shared across variables. DeCo's module omits the sensor embedding s_i (`coord_nonode`); DeCo-S keeps it (`coord`). Without the embedding, the module uses 0.11 million parameters.

**Cross-fitting.** Three precautions keep the held-out targets out of each fold fit: training windows that overlap the held-out target times are dropped; the inner validation set is the last 10% of the remaining windows, with its own overlap rule; and the input filling is causal and recomputed after masking the held-out targets (and, for training inputs, the inner-validation targets). Fold scalers use only the permitted times. Each fold forecaster keeps 82.1–82.4% of the training windows of the final forecaster.

**Training recipe.** First phase (forecasters and the eleven baselines of Table III): MAE over valid nonzero targets in standardized units, Adam with learning rate 1e-3 and weight decay 1e-4, no learning-rate schedule, gradient clipping at 5, effective batch size 16, at most 200 epochs, early stopping on validation MAE with patience 30. Module: AdamW with learning rate 5e-4 and weight decay 1e-4, mini-batches of 512 origin–sensor pairs.

**Second training phase.** Ten epochs from the selected checkpoint with the first phase's optimizer settings (fresh Adam state). At every step, each sensor's rows of the sensor-embedding table and of E1, E2 are replaced, together, by those of a randomly drawn sensor with probability 0.1 (RAGC's embedding swap). The weights are averaged over the ten epochs with equal weights. This applies to the fold, final and early-fusion forecasters.

**Module selection.** The validation score is the MAE over valid nonzero targets in standardized units, averaged over the variables. After every epoch, both deployments are scored (the fold average and the one final forecaster per variable); the uncorrected forecasts (epoch 0) are the first candidate. An epoch counts as an improvement only if it lowers a score by more than 1e-8. Training stops when neither score has improved for eight epochs, or after 25 update epochs. DeCo keeps the best checkpoint under the fold-average score, DeCo-S the best under the single-forecaster score.

**Baselines.** STAEformer has 3 layers and 4 heads; HimNet has hidden size 64 and time embeddings of size 8; D²STGNN has 5 layers of width 32 and embeddings of size 12. RAGC uses its official implementation with 288 steps per day, three inputs and one output. DecoupleST MISO shares its initial weights and batch order with the SISO forecasters. Fits that reached the epoch limit: 23 of 252 external fits, 3 of 36 RAGC fits, 1 of 36 DecoupleST MISO fits, 19 of 468 DecoupleST SISO fold and final fits, and 102 of 582 further-baseline fits. Further-baseline exceptions: DeepAR uses the Gaussian NLL; GTS, HyperD and DSFormer keep their auxiliary losses; AutoSTL keeps its bounded architecture search; DMSTGCN and MVC-STNet use batch size 64; LightGBM uses the L1 objective with up to 1,000 rounds.

**Statistics.** A test window belongs to the day of its first forecast step, which gives 19, 12, 20 and 13 test days on PeMS03, PeMS04, PeMS07 and PeMS08. The intervals resample these days (10,000 draws shared across methods and seeds) and average the seed-level MAEs within each resample. They omit training randomness and are not adjusted for multiple comparisons.

**Traffic states.** A sensor is congested at a step when its speed is at most, and its occupancy at least, a threshold set from the training targets of its dataset, pooled over sensors (`configs/datasets.json`, `python run.py thresholds`). The paper's level S2 uses the 5th percentile of speed and the 95th of occupancy (30.8–53.2 mph, 12.8–25.1%). The looser levels of Fig. 5 are S1 (20th / 80th) and the original definition (35th / 65th), for which PeMS04 keeps its earlier constants, 63.9 mph and 5.98%. Under the original definition, onset and recovery each hold 3.5–4.7% of the valid speed targets, and in the net reduction in speed error obtained by coordinating with all variables rather than the target alone (DeCo-S against the target-only module), onset accounts for 1.8–9.6 times and recovery for 3.5–7.8 times its share of targets; at S2, onset holds 1.0–1.2% of the speed targets and accounts for 5.0–25.5 times that share.

**Cost measurement.** Latency was measured in float32 with TF32 disabled (PyTorch 2.4.1, CUDA 12.1) on one NVIDIA H100: ten warm-up calls, then the median of 100 synchronized calls, from GPU-resident inputs through normalization, the forecasters, the correction, the fold averaging and the blend, on reference checkpoints with the same forward paths. Parameter counts exclude the unused sensor-embedding table. Training hours sum the epoch wall times of the accepted fits, including validation and data access; the missing partial-epoch times of three resumed PeMS07 module runs are estimated from the job records.

**Data.** Some PeMS07 nodes share identical flow records; their station identifiers trace the aligned records and are not an exact station map. A valid entry means that the record has a value.

## Provenance

This repository consolidates implementations from the experiment packages used for the reported runs, with cluster scheduling and internal bookkeeping removed. The PeMS04 and PeMS08 forecasters were trained with an earlier revision of the first-phase code, before fold-local input filling was added for the missing records of PeMS03 and PeMS07; on the retained windows, valid observations are unchanged by filling. Their coordination modules were fitted and selected on float32 labels, which this code keeps for these two datasets (`module_labels` in `configs/datasets.json`); metrics read the float64 arrays. The second training phase, the module without sensor embedding, the blend, the input zeroing and the traffic-state analysis were run as separate packages and are merged into this code.

We checked the consolidated code against the experiment code: the initial weights, outputs and gradients of every forecaster and module class, the ten-epoch second-phase loops and the bank assembly on synthetic data, and, on the real data, the congestion thresholds, the per-seed test metrics of DeCo and the traffic-state table recomputed from the saved forecasts. The public code was not used to retrain any stage on the real data. Device selection was generalized so that the code also runs on CPU; the reported results were produced on GPUs, and the embedding-swap draws differ between CPU and GPU.

## License

The code and the result tables are under the [MIT License](LICENSE). The license does not cover the data, whose sources are listed in `data/README.md`, or the baseline code, which is not redistributed here.
