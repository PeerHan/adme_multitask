#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

data_path = Path("Data/")
ms_path = data_path / "Processed" / "Multitask"
ms_path.mkdir(parents=True, exist_ok=True)

df_path = data_path / "Processed" 
train_df = None
val_df = None
endpoints = ["hPPB", "rPPB", "Sol", "HLM", "RLM", "MDR1_ER"]

for endpoint in endpoints:
    print(f"Current Endpoint : {endpoint}")
    # Previously dropped duplicates
    merged = pd.read_csv(f"Data/Processed/Merged/{endpoint}_merged.csv")

    # Fetch all Dfs
    for dataset in ["Train", "Val", "Test"]:
        intern_path = str(df_path / "Intern" / dataset / f"{endpoint}_chemprop_{dataset.lower()}.csv")
        extern_path =str(df_path / "Extern" / dataset / f"{endpoint}_chemprop_{dataset.lower()}.csv")
        intern_df = pd.read_csv(intern_path)
        extern_df = pd.read_csv(extern_path)
        # Merge Data : Enable same smiles with different tasks
        merged_df = intern_df.merge(extern_df, on="smiles", how="outer", suffixes=("_intern", "_extern")).sample(frac=1, ignore_index=True)
        # Get Datapoints which are in same set (Extern Train, Intern Train etc)
        merged_intersect = merged[(merged.Source_intern == dataset) & (merged.Source_extern == dataset)][["smiles", "activity_intern", "activity_extern"]]
        # remember which smiles
        intersect_smiles = merged_intersect.smiles
        # drop the single task sample 
        merged_df = merged_df[~merged_df.smiles.isin(intersect_smiles)]
        # concat the double task sample if both samples are from the same set
        merged_df = pd.concat((merged_df, merged_intersect), ignore_index=True)
        mt_data_path = ms_path / dataset
        mt_data_path.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(str(mt_data_path / f"{endpoint}_chemprop_{dataset.lower()}.csv"), index=False)

        
    intern_train_df_path = str(df_path / "Intern" / "Train" / f"{endpoint}_chemprop_train.csv")
    intern_val_df_path = str(df_path / "Intern" / "Val" / f"{endpoint}_chemprop_val.csv")
    extern_train_df_path = str(df_path / "Extern" / "Train" / f"{endpoint}_chemprop_train.csv")
    extern_val_df_path = str(df_path / "Extern" / "Val" / f"{endpoint}_chemprop_val.csv")
    
    intern_train_df = pd.read_csv(intern_train_df_path)
    intern_val_df = pd.read_csv(intern_val_df_path)
    intern_train_df = pd.concat((intern_train_df, intern_val_df), ignore_index=True)
    
    extern_train_df = pd.read_csv(extern_train_df_path)
    extern_val_df = pd.read_csv(extern_val_df_path)
    extern_train_df = pd.concat((extern_train_df, extern_val_df), ignore_index=True)
    
    # Create Train-Val and Scheduler-Val for the final evaluation with the same procedure
    multi_task_train_val = intern_train_df.merge(extern_train_df, on="smiles", how="outer", suffixes=("_intern", "_extern")).sample(frac=1, ignore_index=True)
    # Get all samples which are not in any test set
    merged_intersect = merged[(merged.Source_intern != "Test") & (merged.Source_extern != "Test")][["smiles", "activity_intern", "activity_extern"]]
    intersect_smiles = merged_intersect.smiles
    # remove single task smiles which are available as double task smiles (and not in test set)
    multi_task_train_val = multi_task_train_val[~multi_task_train_val.smiles.isin(intersect_smiles)]
    # add them to the train df
    multi_task_train_val = pd.concat((multi_task_train_val, merged_intersect), ignore_index=True)[["smiles", "activity_intern", "activity_extern"]]
    trainval_path = ms_path / "TrainVal" 
    trainval_path.mkdir(parents=True, exist_ok=True)
    multi_task_train_val.to_csv(str(trainval_path / f"{endpoint}_chemprop_train.csv"), index=False)