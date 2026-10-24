from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import os

parent_path = Path(__file__).parent.parent.parent

dataset_name = "BCancerW"
target_col = "diagnosis"

def _read_data():
    df = pd.read_csv(os.path.join(parent_path, f"demo_results\\{dataset_name}.csv"))
    return df

def _export_folds(folds: List[pd.DataFrame]) -> None:
    out_dir = parent_path / "demo_results" / f"{dataset_name}_Kfolds"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, fold in enumerate(folds):
        fold_copy = fold.copy()
        
        cols_to_drop = [
            col for col in fold_copy.columns 
            if str(col).lower() == 'id' or 'unnamed' in str(col).lower()
        ]
        fold_copy.drop(columns=cols_to_drop, inplace=True)
        
        diagnosis_series = fold_copy.pop(target_col).astype(str).str.strip()
        
        fold_copy['target_0_benign'] = (diagnosis_series == 'B').astype(float)
        fold_copy['target_1_malignant'] = (diagnosis_series == 'M').astype(float)
        
        assert fold_copy.shape[1] == 32, f"Attese 32 colonne (30 feat + 2 target), trovate {fold_copy.shape[1]}"
        
        out_file = out_dir / f"{dataset_name}_fold{i}.csv"
        fold_copy.to_csv(out_file, index=False)

def _generate_K_fold_data(dataset: pd.DataFrame, k: int) -> List[pd.DataFrame]:
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset['fold'] = 0
    
    # Assign fold numbers in a stratified way
    # This ensures each fold has (as close as possible) the same number of samples from each class
    for cl in dataset[target_col].unique():
        cl_indices = dataset[dataset[target_col] == cl].index
        dataset.loc[cl_indices, 'fold'] = np.arange(len(cl_indices)) % k
        
    folds = []
    for i in range(k):
        fold_df = dataset[dataset['fold'] == i].drop('fold', axis=1).reset_index(drop=True)
        folds.append(fold_df)
        
    return folds

def refold():
    df = _read_data()
    folds = _generate_K_fold_data(df, k = 5)

    _export_folds(folds)