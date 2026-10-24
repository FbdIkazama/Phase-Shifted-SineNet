import subprocess
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import os
import main.bin.python.data_handling as data_handling

dataset_name = "BCancerW"

base_path = Path(__file__).parent
bin_path = base_path / "main" / "bin" / "src"
resources_path = base_path / "main" / "demo_results"
folds_path = resources_path / f"{dataset_name}_Kfolds"
folds_num = 5
runs_num = 1

""" dataset_path = resources_path / "sin(2x) + cos(3x) - dataset.csv"
loss_path = resources_path / "sin(2x) + cos(3x) - loss.csv"
curve_path = resources_path / "sin(2x) + cos(3x) - curve.csv" """
exe_path = base_path / "PhaseShiftSineNet.exe"
source_path = bin_path / "PhaseShiftSineNet.c"

# Hyperparameters
learn_rate = "0.01"
learn_decay = "0.5"
decay_step = "500"
momentum_beta = "0.9"
rms_beta = "0.999"
E = "0.00000001"

# Net structure parameters
n_layers = "3"
n_node_layers = ["30", "15", "2"]
n_weight_matrix = str(len(n_node_layers) - 1)

# Training parameters
epochs = "2000"
patience = "15"
optimizer = "2" # {Adam : 2, RMSProp : 1, Momentum : 0}
verbose = "0" # {Full text : 1, Short description : 0}

print("C source file compiling...")
# Compiles the source. Overwrites the macro to dinamically handle C code
subprocess.run([
    "gcc", "-O3", str(source_path), "-o", str(exe_path), "-lm",  #-lm is required for math functions
    f"-DLEARNING_RATE={learn_rate}f",
    f"-DLEARNING_DECAY={learn_decay}f",
    f"-DDECAY_STEP={decay_step}",
    f"-DMOMENTUM_BETA={momentum_beta}f",
    f"-DRMS_BETA={rms_beta}f",
    f"-DE={E}f",
    f"-DN_LAYERS={n_layers}",
    f"-DN_NODES_LAYER0={n_node_layers[0]}",
    f"-DN_NODES_LAYER1={n_node_layers[1]}",
    f"-DN_NODES_LAYER2={n_node_layers[2]}",
    f"-DN_WEIGHTS={n_weight_matrix}",
    f"-DEPOCHS={epochs}",
    f"-DPATIENCE={patience}"
], check=True)

def _calc_accuracy() -> np.ndarray:
    acc = []
    for i in range(runs_num):
        curr_acc : float = 0.0
        for j in range(0, folds_num):
            curr_m = pd.read_csv(os.path.join(folds_path, f"Run_{i}", f"{dataset_name}_matrix_fold{j}_run{i}.csv")).to_numpy()
            curr_acc += curr_m.diagonal().sum() / curr_m.sum()
        acc.append(curr_acc/folds_num)
    return np.array(acc)

def _exec() -> None:
    for run in range(runs_num):
        run_dir = os.path.join(folds_path, f"Run_{run}")
        os.makedirs(run_dir, exist_ok=True)
        data_handling.refold()  # Re-calculate the folds on each run to address different dataset combinations

        for i in range(folds_num):
            # Creates the testing dataset with i-th folding
            entry_path = os.path.join(folds_path, f"{dataset_name}_fold{i}.csv")
            if not Path.exists(Path(entry_path)): raise FileNotFoundError

            curr_fold = pd.read_csv(entry_path)

            # Creates the training dataset with all the other foldings
            processed_folds = []
            for j in range(folds_num):
                if j != i:
                    other_entry_path = os.path.join(folds_path, f"{dataset_name}_fold{j}.csv")
                    if not Path.exists(Path(other_entry_path)): raise FileNotFoundError
                    df = pd.read_csv(other_entry_path)
                    processed_folds.append(df)

            training_set = pd.concat(processed_folds, axis=0, ignore_index=True)
            dataset : pd.DataFrame = pd.concat([training_set, curr_fold], axis=0, ignore_index=True)
            
            curr_dataset_path = os.path.join(folds_path, f"{dataset_name}_curr_data.csv")
            dataset.to_csv(curr_dataset_path, index=False)

            # Runs the compiled C file
            loss_path = os.path.join(run_dir, f"{dataset_name}_loss_fold{i}_run{run}.csv")
            matrix_path = os.path.join(run_dir, f"{dataset_name}_matrix_fold{i}_run{run}.csv")
            roc_path = os.path.join(run_dir, f"{dataset_name}_roc_data_fold{i}_run{run}.csv")
            
            subprocess.run([str(exe_path), curr_dataset_path, str(loss_path), str(matrix_path), str(roc_path), str(optimizer), str(verbose)], check=True)

        print(f"\nRun {run} ended!")

def _show_ROC_AUC():
    fig, ax = plt.subplots(figsize=(8, 6))

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for run in range(runs_num):
        run_dir = os.path.join(folds_path, f"Run_{run}")
        
        # Itera su tutti i fold della Cross-Validation
        for fold in range(folds_num):
            roc_file = os.path.join(run_dir, f"{dataset_name}_roc_data_fold{fold}_run{run}.csv")
            df = pd.read_csv(roc_file)
            
            y_true = df['y_true']
            y_prob = df['y_prob_malignant']
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Plotta la curva del singolo fold in grigio chiaro sullo sfondo
            ax.plot(fpr, tpr, color='gray', alpha=0.25, lw=1)
            
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='blue',
            label=rf'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})',
            lw=2.5, alpha=0.9)

    tprs_upper = np.minimum(mean_tpr + 3 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 3 * std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=0.2,
                    label=r'$\pm$ 3 Standard Deviation')

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance (AUC = 0.500)', alpha=0.8)

    ax.set_title('ROC Curve: Breast Cancer Wisconsin (5-Fold CV)', fontsize=16, fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def main() -> None:
    _exec()
    accuracies : np.ndarray = _calc_accuracy()
    print("\n", accuracies, "\n", "Mean accuracy: ", accuracies.mean())
    
    _show_ROC_AUC()

""" loss_file_path = resources_path / f"{func} - loss.csv"
loss_df = pd.read_csv(loss_file_path, header=None, names=["RMSE"])

plt.plot(loss_df.index, loss_df["RMSE"], color='red')
plt.title("SineNet Training Loss")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show() """

if __name__ == "__main__":
    main()