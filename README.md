# SDAP: Similarity-Driven Attention Projections for Class-Incremental Unsupervised Domain Adaptation

Code, per-run results and analysis scripts for the article *Similarity-Driven Attention Projections for Class-Incremental Unsupervised Domain Adaptation* (Gabriel Soares Baptista, Luis Antonio de Souza Junior, Claudine Badue, Alberto F. De Souza, Thiago Oliveira-Santos; Universidade Federal do Espírito Santo).

Every number, table and figure of the article can be recomputed from `analysis/data/` without a GPU. Re-running the experiments needs the datasets and a CUDA GPU.

## Contents

| Folder | Contents |
|---|---|
| `sdap/` | SDAP on frozen DINOv2 ViT-g/14 features: the full method, its ablations, the β sensitivity, the class-imbalance protocol, and PLDCA+AP (PLDCA on the same frozen features and projector). |
| `ci-uda/` | Our re-implementations of ProCA and PLDCA with a ResNet-50 backbone, including PLDCA under the class-imbalance protocol. |
| `analysis/data/` | Raw results: one entry per run (5 runs per task). |
| `analysis/scripts/` | Aggregation, statistical tests (SciPy, statsmodels), tables and figures. |
| `analysis/output/summary.json` | Every aggregated number used in the article. |

## Reproduce the tables and figures

```bash
python3 -m venv .venv
.venv/bin/pip install -r analysis/requirements.txt
.venv/bin/python -I analysis/scripts/run_all.py
```

LaTeX tables are written to `analysis/output/tables/`, EPS figures to `analysis/output/figures/`, and the aggregated values to `analysis/output/summary.json`.

Aggregation: the five runs of each task (source → target pair) are averaged first; the article reports the mean and the sample standard deviation (ddof = 1) of these task means within each dataset. The final accuracy *A* is the accuracy on all classes seen so far after the last increment.

## Result files

Each key of a JSON file is one run, `<source>_<target>_<run>[_<suffix>]`. Each value has:

- `all_data_until_increment_steps[k]`: accuracy on all classes seen up to increment `k` (`acc`, `f1`, `pseudo_label_acc`); *A* is the last `k`;
- `each_step_eval[k]`: accuracy on increment `k` right after training it;
- `re_eval[k]`: accuracy on increment `k` with the final model.

| File | Contents | Produced by |
|---|---|---|
| `sdap.json` | SDAP, all modules, β = 0.1 | `sdap/main.py --table` |
| `sdap_ablation.json` | Ablation; the key suffix lists the **disabled** modules (`il`, `cl`, `triplet`, `dist`) | `sdap/main.py --table --disable` |
| `sdap_imbalance.json` | SDAP under class imbalance, r = 0.1–1.0 | `sdap/main.py --table --imbalance` |
| `sdap_beta.json` | β sensitivity, Office-31 | `sdap/process/train.py`, `train(model, experiment, beta, [])` swept over β |
| `sdap_old_pl.json` | SDAP with the pseudo-labeling of PLDCA | `sdap/process/train_old.py` |
| `pldca_ap.json`, `pldca_ap_imbalance.json` | PLDCA+AP, balanced and under imbalance | `sdap/main.py`, `run_pldca_with_projector()` (`imbalance_gen=True` for the imbalance file) |
| `proca_resnet50.json`, `pldca_resnet50.json` | ProCA and PLDCA, ResNet-50 | `ci-uda/src/experiment/train_proca_pldca_runner.py` |
| `pldca_resnet50_imbalance.json` | PLDCA, ResNet-50, Office-31, r = 0.1–0.9 (r = 1.0 is `pldca_resnet50.json`) | `train_proca_pldca_runner.py --imbalance` |
| `original_reported_accuracy.csv` | Per-task accuracy reported in the ProCA and PLDCA papers, used to check the re-implementations | copied from the papers |

The training scripts write to fixed file names (for example `all_metrics_<backbone>_disabled.json`). The files above were renamed and their run keys normalized when they were collected; their values were not changed.

## Class-imbalance protocol

In every increment, the first class keeps all its target training images and each other class keeps a fraction *r* of them (at least one image). Evaluation always uses all target images of the classes seen so far.

## Datasets

Office-31, Office-Home and ImageNet-Caltech are public benchmarks and are not redistributed here. Both code bases expect a `datasets/` folder with one sub-folder per domain and one per class, for example `datasets/office-31/Amazon/back_pack/*.jpg`. Class orders are in `ci-uda/data/*/config.json` and `sdap/datasets.yml`.

## Run the experiments

### SDAP and PLDCA+AP (`sdap/`)

Run from `sdap/`; the dataset, source and target are set in `experiment.yml`, and the dataset paths in `datasets.yml`.

```bash
cd sdap
pip install -r requirements.txt
python scripts/create_feature_matrix.py   # extract and cache the DINOv2 features in features/
python main.py --table                    # SDAP, all tasks, 5 runs
python main.py --table --disable          # ablation
python main.py --table --imbalance        # imbalance protocol
python main.py --pldca                    # PLDCA+AP
python main.py --table --only-disabled cl triplet dist il \
    --fractions 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 --output results/source_only_imbalance.json  # Source-only under imbalance
```

### ProCA and PLDCA with ResNet-50 (`ci-uda/`)

```bash
cd ci-uda
python -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
export PYTHONPATH=./src:. CIUDA_DATA_ROOT=../datasets
.venv/bin/python scripts/train_sources.py --datasets office-31          # source models in weights/
.venv/bin/python src/experiment/train_proca_pldca_runner.py            # ProCA and PLDCA, balanced
.venv/bin/python src/experiment/train_proca_pldca_runner.py --imbalance --datasets office-31 \
    --fractions 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 --runs 5          # PLDCA under imbalance
```

The imbalance run saves each finished task to `results/pldca_resnet50_imbalance.json` and resumes from it if interrupted. `python3 scripts/progress.py --watch 60` shows its progress. See `ci-uda/README.md` for details.

## License

MIT, see `LICENSE`.
