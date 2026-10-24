from typing import List
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

loss : List[float] = []
n_records = 1
func = "sin(2x) + cos(3x)"

parent_path = Path(__file__).parent.parent

with open(os.path.join(parent_path, f"resources\\{func} - loss.csv"),"r") as f:
    for line in f:
        loss.append(float(line.strip()))
        n_records += 1
plt.figure(figsize=(18, 6))
plt.suptitle(f"PhaseShift Net Performance Analysis: {func} (Adam optimizer - learning decay)", fontsize=15)
plt.subplot(1, 3, 1)
plt.plot(range(1, n_records), loss, color='tab:red')
plt.title(f"PhaseShift Net - {func} validation RMSE")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 2)
df_dataset = pd.read_csv(os.path.join(parent_path, f"resources\\{func} - dataset.csv"))
plt.scatter(df_dataset['X'], df_dataset['Y'], s=2, alpha=0.5, color='tab:blue')
plt.title("Original Dataset Distribution")
plt.xlabel("X value")
plt.ylabel("Y value")
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 3)
df_curve = pd.read_csv(os.path.join(parent_path, f"resources\\{func} - curve.csv"))
plt.plot(df_curve['x_real'], df_curve['target_real'], label=f'Real: {func}', linewidth=2)
plt.plot(df_curve['x_real'], df_curve['pred_real'], label='Predicted (SineNet)', linestyle='--', alpha=0.8)
plt.title(f"Function: {func} - Approximation Comparison")
plt.xlabel("X value")
plt.ylabel("Y value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()