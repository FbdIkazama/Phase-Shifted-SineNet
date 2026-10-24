A custom Deep Learning neural architecture implemented from scratch in pure C.

## Overview
- Low-level implementation without external deep learning libraries.
- Supervised learning with periodic activation functions and phase-shift dynamics.
- Optimized memory layout and cache-friendly operations.

## Benchmark Results (Breast Cancer Wisconsin - 5-Fold CV)
![ROC AUC Curve](main/demo_results/roc_auc_bcancer.png)

## Quickstart & Demo

This repository includes a standalone demonstration pipeline orchestrating the C computational core via Python.

### Prerequisites
- GCC / Clang with C99 support
- Python 3.8+ (`pip install -r requirements.txt`)

### Run the Demo

Clone the repository and launch the demo script:

```bash
# 1. Clone repository
git clone [https://github.com/FbdIkazama/Phase-Shifted-SineNet.git](https://github.com/FbdIkazama/Phase-Shifted-SineNet.git)
cd Phase-Shifted-SineNet

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute demo pipeline
python demo.py
```

The script automatically compiles the C binaries with `-O3` optimization flags and runs an inference/training cycle on sample data located in `data/sample_data.csv`. Outputs (loss curves, confusion matrix, ROC data) are written to `demo_results/`.