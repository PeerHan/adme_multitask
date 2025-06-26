#!/usr/bin/env python3
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import numpy as np
from featurize import standardize

def load_intern_data(
        intern_sdf : str, 
        target : str, 
        endpoint : str, 
        save_folder : str,
        threshold_date : str = "2023-05-01",
    ) -> None:
    """
    Function to load, split (time split) and save internal data (Unprocessed).
    SDF File -> CSV File (saved under save_folder/Intern_train/endpoint_train.csv)
    This function integrates an train/val/test split method based on time
   
    Parameters:
        - intern_sdf : SDF File which will be splitted in Train, Val, Test data based on time splits
        - target : Target_col for the endpoint - the target will be renamed to "activity"
        - endpoint : Naming the saved file
        - save_folder : Path to the top level directory of the unprocessed internal data
        - threshold_date : Threshold to split Train-val from Test set
    """
    
    df = PandasTools.LoadSDF(intern_sdf)
    df = df.drop_duplicates("SMILES", keep="last")
    df.EXPERIMENT_DATE = pd.to_datetime(df.EXPERIMENT_DATE)
    
    time_series = df[df.EXPERIMENT_DATE < threshold_date]
    time_series = (time_series.EXPERIMENT_DATE.sort_values().value_counts().sort_index() / len(time_series)).iloc[::-1].cumsum()
    val_split = time_series[time_series <= 0.2].index.min()
    
    test = df[df.EXPERIMENT_DATE >= threshold_date]
    val = df[(df.EXPERIMENT_DATE >= val_split) & (df.EXPERIMENT_DATE < threshold_date)]
    train = df[df.EXPERIMENT_DATE < val_split]
    
    print(f"Splits on : Train - {train.EXPERIMENT_DATE.min()}, {train.EXPERIMENT_DATE.max()}")
    print(f"Splits on : Val - {val.EXPERIMENT_DATE.min()}, {val.EXPERIMENT_DATE.max()}")
    print(f"Splits on : Test - {test.EXPERIMENT_DATE.min()}, {test.EXPERIMENT_DATE.max()}")
    
    
    # Drop duplicates
    train_smiles = train.SMILES
    val_smiles = val.SMILES
    duplicates_test_train = test[test.SMILES.isin(train_smiles)].SMILES
    duplicates_test_val = test[test.SMILES.isin(val_smiles)].SMILES
    duplicates_val_train = val[val.SMILES.isin(train_smiles)].SMILES
    
    if len(duplicates_test_train) > 0:
        print(f"Dropping {len(duplicates_test_train)} duplicates from train because of intersections in test-train")
        train = train[~train.SMILES.isin(duplicates_test_train)]
    if len(duplicates_test_val) > 0:
        print(f"Dropping {len(duplicates_test_val)} duplicates from val because of intersections in test-val")
        val = val[~val.SMILES.isin(duplicates_test_val)]
    if len(duplicates_val_train) > 0:
        print(f"Dropping {len(duplicates_val_train)} duplicates from train because of intersections in val-train")
        train = train[~train.SMILES.isin(duplicates_val_train)]
        
        
    
    assert train.EXPERIMENT_DATE.max() < val.EXPERIMENT_DATE.min() and val.EXPERIMENT_DATE.max() < test.EXPERIMENT_DATE.min(), "Timesplit : The oldest test date should be more recent than the newest train date"
    assert len(set(train.SMILES).intersection(set(test.SMILES))) == 0, "Duplicates : Train and Testset should not share common SMILES"
    assert len(set(train.SMILES).intersection(set(val.SMILES))) == 0, "Duplicates : Train and Valset should not share common SMILES"
    assert len(set(val.SMILES).intersection(set(test.SMILES))) == 0, "Duplicates : Val and Testset should not share common SMILES"
    
    # Rename to ensure same naming conventions between extern/intern data
    # Keep Experiment Date in the train for a train/val split
    train = train.rename({"SMILES" : "smiles", target : "activity", "EXPERIMENT_DATE" : "date", "RESULT_UNIT" : "unit", "OPERATOR" : "operator"}, axis=1)
    val = val.rename({"SMILES" : "smiles", target : "activity", "EXPERIMENT_DATE" : "date", "RESULT_UNIT" : "unit", "OPERATOR" : "operator"}, axis=1)
    test = test.rename({"SMILES" : "smiles", target : "activity", "EXPERIMENT_DATE" : "date", "RESULT_UNIT" : "unit", "OPERATOR" : "operator"}, axis=1)
    
    # Reset numerical index
    test = test.reset_index(drop=True)
    val = val.reset_index(drop=True)
    train = train.reset_index(drop=True)
    
    # Save train/test data
    save_cols = ["smiles", "activity", "unit", "operator"]
    train[save_cols + ["date"]].to_csv(f"{save_folder}/Intern_Train/{endpoint}_train.csv", index=False)
    val[save_cols + ["date"]].to_csv(f"{save_folder}/Intern_Val/{endpoint}_val.csv", index=False)
    test[save_cols + ["date"]].to_csv(f"{save_folder}/Intern_Test/{endpoint}_test.csv", index=False)
    
    return None

def train_val_split(
        train : "DataFrame", 
        val_fraction : float = 0.2,
    ) -> ["DataFrame", "DataFrame"]:
    """
    Split an existing training set into train and validation set.
    val fraction determines how many train samples are used for the validation set.
    the split will be determined with a timesplit 
    train -> [train, val]
    
    Parameters:
        - train : Training data which should be split into train and val
        - val_fraction : Regulate the size of validation set - using the relative frequency based on a timesplit which exceeds 0.2 of the total data
    """
    # Assert datetime
    train.date = pd.to_datetime(train.date)
    
    # Create a series : Index as year, Value as relativ frequency (data)
    time = (train.sort_values("date").date.apply(lambda time_stamp: time_stamp.year).value_counts().sort_index() / train.shape[0]).iloc[::-1].cumsum()
    # Get the year which exceeds the threshold
    split_year = time[time > val_fraction].index[0]
    # Get the relativ frequency
    relativ_val_size = time[split_year]
    
    assert 0.2 <= relativ_val_size <= 0.35, "Split per Year : Validation set is to tiny/big"
    
    # Split and return new sets for further proccesing
    # Dropping the date col (not needed anymore) and resetting the index
    val = train[train.date >= str(split_year)].drop("date", axis=1).reset_index(drop=True)
    train = train[train.date < str(split_year)].drop("date", axis=1).reset_index(drop=True)
    
    return train, val

def transform_target(
        df : "DataFrame", 
        target_col : str, 
        transformation : "AggFuncType",
    ) -> "DataFrame":
    """
    Make a copy of a dataframe (df) and transform the target col
    df -> df
    
    Parameters:
        - df : Dataframe with the col to transform
        - target_col : Name of the target
        - transformation : Aggregation function to transform the target
    """
    copy_df = df.copy()
    copy_df[target_col] = copy_df[target_col].transform(transformation)
    return copy_df

def log_inverse(
        array : "Series", 
        base : int = 10,
    ) -> "Series":
    """
    Inverse function for log which can be applied as a pandas aggregation function
    """
    return base ** array

def scaffold_split(
        train_df : "DataFrame", 
        val_size : float = 0.2,
    ) -> ["DataFrame", "DataFrame"]:
    """
    Train -> [Train, Val]
    Splitting a dataframe with a scaffold split (Bermis-Murcko scaffold)
    Train and validation sets wont share any scaffold.
    
    Parameters:
        - Train df requires a "Smile" column named "smiles"
        - val_size regulates the size of the validation set (0.2 * n per default)
    """
    
    # Generate scaffolds
    PandasTools.AddMoleculeColumnToFrame(train_df, "smiles")
    train_df["scaffold"] = train_df.smiles.apply(MurckoScaffold.MurckoScaffoldSmiles)
    
    # Relativ scaffold frequencies -> sample with frac 1 to shuffle the scaffolds -> cumsum to see cumulated frequencies
    scaffold_frequencies = train_df.scaffold.value_counts() / train_df.shape[0]
    cummulated_randomized_frequencies = scaffold_frequencies.sample(frac=1.0).cumsum()
    
    # Unique scaffolds for train and val set
    train = []
    val = []
    for scaffold, freq in cummulated_randomized_frequencies.items():
        
        if freq > val_size:
            train.append(scaffold)
        else:
            val.append(scaffold)
    
    # Filter per unique scaffold
    val_df = train_df[train_df.scaffold.isin(val)]
    train_df = train_df[train_df.scaffold.isin(train)]
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    return train_df, val_df

def sol_transformation(
        smile : str,
        activity : float,
    ) -> float:
    """
    Transform solubility from log10(ug/ml) -> log10(mol/l)
    The input should be a row which is suitable for a lambda expression in a pandas aggregation function
    e.g. df.apply(lambda row : sol_transformation(row.smiles, row.activity), axis=1)
    """
    weights = ExactMolWt(Chem.MolFromSmiles(smile))
    activity = activity - 3 - np.log10(weights)
    return activity