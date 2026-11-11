# Data

Each dataset has its own folder, `data/<dataset>/`, with these arrays (T steps, N sensors, 3 variables in the order flow, occupancy, speed):

| File | Shape and type | Content |
|---|---|---|
| `labels.npy` | T×N×3, float64 | Raw values, finite everywhere (invalid entries hold 0). Occupancy is a fraction; speed is in mph. |
| `valid.npy` | T×N×3, bool | Target-validity mask. |
| `causal.npy` | T×N×3, float32 | Causal forward fill of the valid values, with zeros before the first valid value. |
| `adj.npy` | N×N, float32 | Road-graph adjacency (1 where the source graph has an edge), used only by D²STGNN. |

Dataset sizes, splits and congestion thresholds are in `../configs/datasets.json`. `MANIFEST.json` records checksums for the released PeMS03 and PeMS07 arrays and archives, and for the upstream PeMS04 and PeMS08 `.npz` files.

## PeMS03 and PeMS07

The aligned arrays are in the [`data-v1` release](https://github.com/lixus7/DeCo/releases/tag/data-v1). From this folder:

```bash
curl -LO https://github.com/lixus7/DeCo/releases/download/data-v1/pems03.tar.gz
curl -LO https://github.com/lixus7/DeCo/releases/download/data-v1/pems07.tar.gz
curl -LO https://github.com/lixus7/DeCo/releases/download/data-v1/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf pems03.tar.gz
tar -xzf pems07.tar.gz
```

These are the arrays the reported PeMS03 and PeMS07 runs used.

**How they were made.**

- **Flow** is copied unchanged from the PeMS03 and PeMS07 benchmark files distributed with [STSGCN](https://github.com/Davidham3/STSGCN).
- **Occupancy and speed** come from the five-minute station records of the [Caltrans PeMS](https://pems.dot.ca.gov) data clearinghouse for the same periods. PeMS03 sensors follow the published station list (`PEMS03.txt`, next to the edge list below). PeMS07's benchmark nodes are anonymous, so they were matched by comparing their flow with the PeMS records over all 98 days. Five nodes with incomplete records were matched on shorter periods or by their place in the graph, and where candidates differed in occupancy or speed (40 nodes), the match also used the benchmark graph and PeMS station coordinates.
- **Validity:** flow is valid everywhere. An occupancy or speed entry is valid if the record has a value there; otherwise it is invalid and holds 0 in `labels.npy`.
- **Graph:** PeMS03 uses the edge list distributed with [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS03), and PeMS07 uses the benchmark's edge list, keeping its node order.

| | PeMS03 | PeMS07 |
|---|---|---|
| Size (T×N) | 26,208×358 | 28,224×883 |
| Period | 2018-09-01 to 2018-11-30 | 2017-05-01 to 2017-08-06 |
| Invalid occupancy / speed | 0.48% / 2.62% | 0.20% / 1.25% |
| Sensors with no valid speed | 8 | 11 |
| PeMS flow within 1e-6 of the benchmark flow, where PeMS has a value | 99.956% of cells | 100% of cells |

**How to read them.**

- These arrays are PeMS records aligned to the flow benchmarks.
- A valid entry means that the record has a value.
- Where PeMS03's records differ from the benchmark flow, the benchmark value is kept.

**Provenance** (`provenance/`):

- `<dataset>_sources.csv`: name, SHA-256 and size of every PeMS station file used.
- `<dataset>_station_ids.txt`: the PeMS station ID of each sensor, in array order.

PeMS07's benchmark nodes are anonymous and some share a complete flow series, so some of its station IDs represent a set of stations with identical records. Use them to trace the records, not as an exact station map, and keep the node-index graph.

## PeMS04 and PeMS08

These use the three-variable releases in [ASTGCN](https://github.com/Davidham3/ASTGCN-2019-mxnet) at commit `d5d40645fa45471f2b473fc49a29ec3f0a993c6d`: `data/PEMS04/` and `data/PEMS08/`, each with an `.npz` file and `distance.csv`. To build the arrays, from the repository root:

```bash
python data/make_pems0408.py --npz pems04.npz --distance distance.csv --out data/pems04
```

All finite entries in these releases are treated as valid.

## License and attribution

This repository's MIT License does not cover the data, and the repository grants no separate data license. Occupancy and speed come from Caltrans PeMS, and flow and the graphs from the releases linked above, so the terms of those sources apply. The linked benchmark repositories declare no license, and the [PeMS conditions of use](https://pems.dot.ca.gov/?view=tou) are a general site-use policy, so check these sources before redistributing the arrays. Please credit them when you use the arrays.
