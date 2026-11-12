# ci-uda

Re-implementation of ProCA/PLDCA with a ResNet-50 backbone.

```bash
export PYTHONPATH=./src:.:$PYTHONPATH
# Optional: root that contains the dataset folders (default: ./datasets)
export CIUDA_DATA_ROOT=/path/to/dataset-root
```

Train the Office-31 source models (`weights/<domain>_source.pth`):

```bash
python scripts/train_sources.py --datasets office-31
```

PLDCA under the class-imbalance protocol (resumes by skipping keys already in the JSON):

```bash
python src/experiment/train_proca_pldca_runner.py --imbalance --datasets office-31 \
    --fractions 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 --runs 5
```

Tiny end-to-end smoke test (separate JSON, never mixed with real results):

```bash
python src/experiment/train_proca_pldca_runner.py --smoke
```

Progress and ETA of a running imbalance experiment (system Python, no venv needed):

```bash
python3 scripts/progress.py --watch 60
```
