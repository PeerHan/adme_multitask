from optuna_toolkit import *
import joblib
import optuna
from pathlib import Path
import pandas as pd
import warnings
import argparse
warnings.simplefilter(action='ignore', category=FutureWarning)
from optuna.samplers import TPESampler

parser = argparse.ArgumentParser()
parser.add_argument('--endpoint', default='hPPB', type=str)
parser.add_argument('--dataset', default="all", type=str)
parser.add_argument('--trials', default=100, type=int)
endpoint = vars(parser.parse_args())["endpoint"]
ds = vars(parser.parse_args())["dataset"]
trials = vars(parser.parse_args())["trials"]

data_path = Path("Data/")
log_path = data_path / "HP_Results" / "Logs"
result_path = data_path / "HP_Results" / "ML"
log_path.mkdir(parents=True, exist_ok=True)
result_path.mkdir(parents=True, exist_ok=True)

regressors = {
    "XGB" : {
        "objective" : xgb_objective,
        "trials" : trials
    },
    "Chemprop" : {
        "objective" : chemprop_objective,
        "trials" : trials
    },
    "MLP" : {
        "objective" : mlp_objective, 
        "trials" : trials,
    },
}

optuna.logging.set_verbosity(optuna.logging.WARNING)

print(endpoint, flush=True)
assert endpoint in ["hPPB", "rPPB", "Sol", "HLM", "RLM", "MDR1_ER"]
assert ds in ["all", "extern", "intern", "inex", "intern+", "extern+"]
df_path = data_path / "Processed" 
train_df = None
val_df = None
print("HP Search started", flush=True)
print("Current Endpoint : " + endpoint, flush=True)

ds_map = {"all" : ["Extern", "Intern", "InEx"],
          "extern" : ["Extern"],
          "intern" : ["Intern"],
          "inex" : ["InEx"],
          "intern+" : ["Intern", "InEx"],
          "extern+" : ["Extern", "InEx"]}

for df_comb in ds_map[ds]:
    print("Current Trainingset : " + df_comb, flush=True)

    for regressor, reg_params in regressors.items():
        file = "_processed_" if regressor != "Chemprop" else "_chemprop_"
        train_df_path = str(df_path / df_comb / "Train" / (endpoint + file + "train.csv"))
        val_df_path = str(df_path / df_comb / "Val" / (endpoint + file + "val.csv"))
        train_df = pd.read_csv(train_df_path)
        val_df = pd.read_csv(val_df_path)
        # Max R2, Min MAE
        log_endpoint_path = log_path / endpoint
        log_endpoint_path.mkdir(parents=True, exist_ok=True)
        log_path_endpoint = str(log_endpoint_path / (regressor + "_" + df_comb + "_LOGS.txt"))
        print("Starting Optuna with Regressor : " + regressor, flush=True)
        study = optuna.create_study(directions=["maximize", "minimize"], study_name=regressor, sampler=TPESampler())
        study.optimize(lambda trial : reg_params["objective"](trial, train_df, val_df, log_path=log_path_endpoint), 
                       n_trials=reg_params["trials"],
                       n_jobs=1,
                       timeout=None,
                       show_progress_bar=False)
        model_res_path = result_path / endpoint 
        model_res_path.mkdir(parents=True, exist_ok=True)
        study_df = study.trials_dataframe()
        study_df = study_df.rename({"values_0" : "R2", "values_1" : "MAE"}, axis=1)
        study_df.to_csv(str(result_path) + "/" + endpoint + "/" + regressor + "_" + df_comb + "_hp_df.csv", index=False)
        joblib.dump(study, str(model_res_path) + "/" + regressor + "_" +  df_comb + ".pkl")