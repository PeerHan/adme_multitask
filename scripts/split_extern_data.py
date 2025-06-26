from data import scaffold_split
from pathlib import Path
import pandas as pd
import os

data_path = Path("Data/Unprocessed/")

unsplit = data_path / "Extern_Unsplit"
train_path = data_path / "Extern_Train"
val_path = data_path / "Extern_Val"
target = "activity"
train_path.mkdir(parents=True, exist_ok=True)
val_path.mkdir(parents=True, exist_ok=True)

endpoint_data = {
    "hPPB_train" : {
        "Alias" : "hPPB"
    },
    "rPPB_train" : {
        "Alias" : "rPPB"
    },
    "Sol_train" : {
        "Alias" : "Sol"
    },
    "HLM_train" : {
        "Alias" : "HLM"
    },
    "RLM_train" : {
        "Alias" : "RLM"
    }
}

for endpoint in ["hPPB", "rPPB", "Sol", "HLM", "RLM", "MDR1_ER"]:

    save_train_path = str(train_path / f"{endpoint}_train.csv")
    save_val_path = str(val_path / f"{endpoint}_val.csv")
    unsplit_file = str(unsplit / f"{endpoint}_train.csv")
    
    train_df = pd.read_csv(unsplit_file)
    train_df, val_df = scaffold_split(train_df)
    
    train_df.to_csv(save_train_path, index=False)
    val_df.to_csv(save_val_path, index=False)