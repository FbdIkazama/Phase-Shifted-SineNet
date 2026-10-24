from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

loss : List[float] = []
n_records = 1
func = "Iris_oneHot"

parent_path = Path(__file__).parent.parent

with open(os.path.join(parent_path, f"resources\\{func} - loss.csv"),"r") as f:
    for line in f:
        loss.append(float(line.strip()))
        n_records += 1
plt.figure(figsize=(18, 6))
plt.suptitle(f"PhaseShift Net Performance Analysis: {func} (Adam optimizer - learning decay)", fontsize=15)

plt.subplot(1, 2, 1)
plt.plot(range(1, n_records), loss, color='tab:red')
plt.title(f"PhaseShift Net - {func} validation CE")
plt.xlabel("Epochs")
plt.ylabel("CE")
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
df_cm = pd.read_csv(os.path.join(parent_path, f"resources\\{func} - curve.csv"))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Setosa", "Versicolor", "Virginica"], 
            yticklabels=["Setosa", "Versicolor", "Virginica"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.show()