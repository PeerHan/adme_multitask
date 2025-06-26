#!/usr/bin/env python3

from pathlib import Path
from data import sol_transformation
from numpy import log10
import pandas as pd
from featurize import create_featurized_df, standardize
from rdkit.Chem import MolFromSmiles
import numpy as np

# paths to data folder
data_path = Path("Data")
unprocessed_data = data_path / "Unprocessed"
processed_data = data_path / "Processed"

processed_data.mkdir(parents=True, exist_ok=True)

# config for transformation per endpoint
endpoint_transformations = {
    "hPPB" : log10,
    "rPPB" : log10,
    "Sol" : sol_transformation,
    "HLM" : log10,
    "RLM" : log10,
    "MDR1_ER" : log10
}

# subfolders in Data/Processed/
# Extern : Extern_Train, ... 
# Intern : Intern_Train, ...
folder_combinations = [prefix + suffix for prefix in ["Extern_", "Intern_"] for suffix in ["Train", "Val", "Test"]]

# For Each Endpoint with the corresponding transformation
for endpoint, transformation in endpoint_transformations.items():
    
    print(f"Current endpoint : {endpoint}")
    # First : Drop all Duplicates of standardized SMILES in Intern
    intern_data = []
    extern_data = []
    for suffix in ["Train", "Val", "Test"]:
        extern_unprocessed_df = pd.read_csv(str(unprocessed_data / f"Extern_{suffix}" / f"{endpoint}_{suffix.lower()}.csv"))
        intern_unprocessed_df = pd.read_csv(str(unprocessed_data / f"Intern_{suffix}" / f"{endpoint}_{suffix.lower()}.csv"))
        extern_unprocessed_df["Source"] = suffix
        intern_unprocessed_df["Source"] = suffix      
        print(f"Standardizing SMILES ... ({suffix})")
        intern_unprocessed_df["smiles"] = intern_unprocessed_df.smiles.apply(lambda smiles : standardize(smiles))
        extern_unprocessed_df["smiles"] = extern_unprocessed_df.smiles.apply(lambda smiles : standardize(smiles))
        intern_data.append(intern_unprocessed_df)
        extern_data.append(extern_unprocessed_df)

    intern_data = pd.concat(intern_data, ignore_index=True)
    extern_data = pd.concat(extern_data, ignore_index=True)
    print(f"Intern Size before Drop : {len(intern_data)}; Extern Size before Drop : {len(extern_data)}")
    extern_data = extern_data.drop_duplicates("smiles", keep="last")
    intern_data = intern_data.drop_duplicates("smiles", keep="last")
    print(f"Intern Size after Drop internal : {len(intern_data)}; Extern Size after Drop internal : {len(extern_data)}")

    
    # All Duplicates between intern and extern
    duplicates = intern_data[intern_data.smiles.isin(extern_data.smiles)].smiles
    # Save Duplicates for Multitask
    duplicate_df = intern_data[intern_data.smiles.isin(duplicates)]
    duplicate_df = duplicate_df.merge(extern_data, on="smiles", how="inner", suffixes=("_intern", "_extern"))
    if endpoint == "Sol":
        duplicate_df.activity_extern = duplicate_df.apply(lambda row : sol_transformation(row.smiles, row.activity_extern), axis=1)
    else:
        duplicate_df.activity_intern = duplicate_df.activity_intern.transform(np.log10)
    merged_data_path = processed_data / "Merged"
    merged_data_path.mkdir(parents=True, exist_ok=True)
    duplicate_df.to_csv(str(merged_data_path / f"{endpoint}_merged.csv"), index=False)
    duplicates.to_csv(str(merged_data_path / f"{endpoint}_duplicates.csv"), index=False)
    # Drop all Dup in intern
    intern_data = intern_data[~intern_data.smiles.isin(duplicates)]
    
    print(f"Intern Size after Drop : {len(intern_data)}; Extern Size after Drop : {len(extern_data)}; Duplicates : {len(duplicates)}")

    # Save Unprocessed version but with dropped duplicates 
    for suffix in ["Train", "Val", "Test"]:
        intern_df = intern_data[intern_data.Source == suffix].reset_index(drop=True)
        extern_df = extern_data[extern_data.Source == suffix].reset_index(drop=True)
        extern_df.to_csv(str(unprocessed_data / f"Extern_{suffix}" / f"{endpoint}_{suffix.lower()}_dropped.csv"), index=False)
        intern_df.to_csv(str(unprocessed_data / f"Intern_{suffix}" / f"{endpoint}_{suffix.lower()}_dropped.csv"), index=False)

    # For each dataset (Train/Val/Test X Intern/Extern)
    for split_folder in folder_combinations:
        
        # Path to current dataset
        split_path = unprocessed_data / split_folder
        
        # Infer the suffix from the folder name
        folder_set, dataset = split_folder.split("_")
        suffix = "_" + dataset.lower() + ".csv"
        
        # Path to the file
        endpoint_file = str(split_path / f"{endpoint}_{dataset.lower()}_dropped.csv")
        
        # Transform the target and save + standardize SMILES
        df = pd.read_csv(endpoint_file)
        print(f"reading file {endpoint_file}")
            
        # Transform target cols
        # Schedule for Intern Data - log10 for hPPB/rPPB/RLM/HLM/MDR1-ER
        if folder_set == "Intern" and endpoint != "Sol":
            df.activity = df.activity.transform(transformation)

        # Schedule for Extern Data - Transform Extern Sol
        elif folder_set == "Extern" and endpoint == "Sol":
            df.activity = df.apply(lambda row : sol_transformation(row.smiles, row.activity), axis=1)
        
        # Save df for chemprop : Smiles col and Target
        chemprop_data_path = processed_data / folder_set / dataset
        chemprop_data_path.mkdir(parents=True, exist_ok=True)
        chemprop_path = str(chemprop_data_path / f"{endpoint}_chemprop{suffix}")
        smile_df = df[["smiles", "activity"]]
        smile_df.to_csv(chemprop_path, index=False)
        # Featurize and save for other ML : 1340 descriptors
        tabular_data_path = processed_data / folder_set / dataset
        tabular_data_path.mkdir(parents=True, exist_ok=True)
        save_path = str(tabular_data_path / f"{endpoint}_processed{suffix}")
        featurized_df = create_featurized_df(smile_df)            
        featurized_df.to_csv(save_path, index=False)
        
    # Schedule for Mixing Intern and Extern Data
    for dataset in ["Train", "Val"]:
        intern_folder = processed_data / "Intern" / dataset
        extern_folder = processed_data / "Extern" / dataset
        suffix = "_" + dataset.lower() + ".csv"
        for model_suffix in ["_processed", "_chemprop"]:
            intern_endpoint = str(intern_folder / f"{endpoint}{model_suffix}{suffix}")
            extern_endpoint = str(extern_folder / f"{endpoint}{model_suffix}{suffix}")
            intern_df = pd.read_csv(intern_endpoint)
            extern_df = pd.read_csv(extern_endpoint)
            merged_df = pd.concat((intern_df, extern_df), axis=0)
            inex_data_path = processed_data / "InEx" / dataset 
            inex_data_path.mkdir(parents=True, exist_ok=True)
            save_path = str(inex_data_path/ f"{endpoint}{model_suffix}{suffix}")
            merged_df = merged_df.reset_index(drop=True)
            merged_df.to_csv(save_path, index=False)
    print()