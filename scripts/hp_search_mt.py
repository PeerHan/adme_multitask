#!/usr/bin/env python3

from optuna_toolkit import *
import joblib
import optuna
from pathlib import Path
import pandas as pd
import warnings
from optuna.samplers import TPESampler
import argparse
warnings.simplefilter(action='ignore', category=FutureWarning)


data_path = Path("Data/")
log_path = data_path / "HP_Results" / "Logs"
result_path = data_path / "HP_Results" / "ML"
log_path.mkdir(parents=True, exist_ok=True)
result_path.mkdir(parents=True, exist_ok=True)

regressor = "Chemprop_MT"
objective = chemprop_objective
optuna.logging.set_verbosity(optuna.logging.WARNING)
parser = argparse.ArgumentParser()
parser.add_argument('--endpoint', default='hPPB', type=str)
parser.add_argument('--trials', default=100, type=int)
trials = vars(parser.parse_args())["trials"]
endpoint = vars(parser.parse_args())["endpoint"]
print(endpoint, flush=True)
assert endpoint in ["hPPB", "rPPB", "Sol", "HLM", "RLM", "MDR1_ER"]
df_path = data_path / "Processed" 
train_df = None
val_df = None
print("HP Search started", flush=True)
print("Current Endpoint : " + endpoint, flush=True)

train_df_path = str(df_path / "Multitask" / "Train" / (endpoint + "_chemprop_train.csv"))
val_df_path = str(df_path / "Multitask" / "Val" / (endpoint + "_chemprop_val.csv"))
train_df = pd.read_csv(train_df_path)
val_df = pd.read_csv(val_df_path)

# Max R2, Min MAE
log_path_endpoint = str(log_path / endpoint / (regressor + "_LOGS.txt"))
print("Starting Optuna with Regressor : " + regressor, flush=True)
study = optuna.create_study(directions=["maximize", "minimize"], study_name=regressor, sampler=TPESampler())
study.optimize(lambda trial : objective(trial, train_df, val_df, log_path=log_path_endpoint, n_tasks=2, target_columns=["activity_intern", "activity_extern"]), 
               n_trials=trials,
               n_jobs=1,
               timeout=None,
               show_progress_bar=False)

model_res_path = result_path / endpoint 
model_res_path.mkdir(parents=True, exist_ok=True)
study_df = study.trials_dataframe()
study_df = study_df.rename({"values_0" : "R2", "values_1" : "MAE"}, axis=1)
study_df.to_csv(str(result_path) + "/" + endpoint + "/" + regressor + "_hp_df.csv", index=False)
joblib.dump(study, str(model_res_path) + "/" + regressor + ".pkl")