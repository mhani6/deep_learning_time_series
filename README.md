# PatchTST — LSST Time Series Classification

Deep Learning for Time Series project — Setting 1 (foundation model adaptation).

## Setup local (VS Code)

```bash
pip install -r requirements.txt
```

## Run

```bash
# Train all models (PatchTST + UniTS + baseline)
python train.py --model all

# Individual models
python train.py --model patchtst --epochs 150
python train.py --model units    --epochs 150
python train.py --model baseline
```


## Project structure

```
├── data/
│   └── dataset.py        # LSST loading, normalization, DataLoader
├── models/
│   ├── patchtst.py       # PatchTST classifier (main model)
│   └── baseline.py       # InceptionTime baseline
├── utils/
│   ├── trainer.py        # Training loop, early stopping
│   └── evaluate.py       # Metrics, classification report
├── runs/                 # Checkpoints and results (generated)
├── train.py              # Main entry point
├── colab_run.ipynb       # Colab notebook for GPU runs
└── requirements.txt
```

## Model summary

| Model | Type | Key idea | Paper |
|---|---|---|---|
| PatchTST | Transformer (patch-based) | Series as patch tokens, channel-independent | ICLR 2023 |
| UniTS | Unified Transformer (seq + var attention + DLO) | Task tokenization, dual attention | NeurIPS 2024 |
| InceptionTime | CNN | Multi-scale inception blocks | 2020 |

## Teammates

- Your name — PatchTST
- Teammate 1 — MOMENT
- Teammate 2 — CHRONOS
