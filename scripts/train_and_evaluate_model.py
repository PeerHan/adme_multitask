from pathlib import Path
import pandas as pd
from torch_toolkit import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from optuna_toolkit import create_optuna_mlp, best_trial_mlp, best_trial_xgb, best_trial_chemprop, create_optuna_chemprop
import xgboost
from math import sqrt
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
import pickle
from Evaluation import get_metrics
import argparse
from lightning import pytorch as pl
import os


parser = argparse.ArgumentParser()
parser.add_argument('--endpoint', default='hPPB', type=str)
parser.add_argument('--models', default='all', type=str)
parser.add_argument('--datasets', default='all', type=str)
parser.add_argument('--n_evals', default=30, type=int)
args = vars(parser.parse_args())
endpoint = args["endpoint"]
models = args["models"]
datasets = args["datasets"]
print(endpoint, flush=True)
assert endpoint in ["hPPB", "rPPB", "Sol", "HLM", "RLM", "MDR1_ER"]
assert models in ["all", "Chemprop", "XGB", "MLP"]
assert datasets in ["all", "Intern", "InEx", "MT", "Extern"]
repetitions = args["n_evals"]
device = "cuda"

model_map = {"all" : ["XGB", "Chemprop", "MLP"],
             "Chemprop" : ["Chemprop"],
             "MLP" : ["MLP"],
             "XGB" : ["XGB"]}

dataset_map = {"all" : ["Extern", "Intern", "InEx", "MT"],
               "Intern" : ["Intern"],
               "InEx" : ["InEx"],
               "MT" : ["MT"],
               "Extern" : ["Extern"]}

data_path = Path("Data/")
print("Current Endpoint : " + endpoint, flush=True)

# Evaluate all training strategies
for dataset in dataset_map[datasets]:
    # Evaluate each model
    for model in model_map[models]:
        if model != "Chemprop" and dataset == "MT":
            continue
        eval_path = data_path / "Evaluation" / model / dataset
        eval_path.mkdir(parents=True, exist_ok=True)
        conf_folder = data_path / "Confidence" / endpoint / model / dataset
        conf_folder.mkdir(parents=True, exist_ok=True)

        model_type = "_chemprop_" if model == "Chemprop" else "_processed_"
        print("Current Dataset : " + dataset, flush=True)
        # Multitask folder as case handel
        if dataset == "MT":
            dataset_path = data_path / "Processed" / "Multitask"
            train_val_df = pd.read_csv(str(dataset_path / "TrainVal" / (endpoint + "_chemprop_train.csv")))
            test_df = pd.read_csv(str(dataset_path / "Test" / (endpoint + "_chemprop_test.csv")))
        # For Single Chemprop, XGB and MLP use singletask data
        else:
            dataset_path = data_path / "Processed" / dataset
            # Concat Train and Val data to not vaste any data
            train_data = str(dataset_path / "Train" / (endpoint + model_type + "train.csv"))
            val_data = str(dataset_path / "Val" / (endpoint + model_type + "val.csv"))
            test_intern_data = str(data_path / "Processed" / "Intern" / "Test" / (endpoint + model_type + "test.csv"))
            test_extern_data = str(data_path / "Processed" / "Extern" / "Test" / (endpoint + model_type + "test.csv"))

            train_df = pd.read_csv(train_data)
            val_df = pd.read_csv(val_data)
            test_df_intern = pd.read_csv(test_intern_data)
            test_df_extern = pd.read_csv(test_extern_data)
            train_val_df = pd.concat((train_df, val_df), axis=0).reset_index(drop=True)

        print("Current Model : " + model, flush=True)
        # Define paths for the optuna-df and loggings (epochs)
        trial_path = str(data_path / "HP_Results" / "ML" / endpoint / (model + "_" + dataset + "_hp_df.csv"))
        trial_df = pd.read_csv(trial_path)

        # Finding best training run
        best_intern_preds = None
        best_extern_preds = None
        min_mae_intern = float("inf")
        min_mae_extern = float("inf")
        # Neural Network schedule
        if model == "MLP":
            train_val_set = MixedDescriptorSet(train_val_df)
            intern_test_loader = DataLoader(MixedDescriptorSet(test_df_intern), batch_size=len(test_df_intern), shuffle=False)
            extern_test_loader = DataLoader(MixedDescriptorSet(test_df_extern), batch_size=len(test_df_extern), shuffle=False)
            mlp_results = {}
            log_path = str(data_path / "HP_Results" / "Logs" / endpoint / f"MLP_{dataset}_LOGS.txt")
            trial_data = best_trial_mlp(trial_df, log_path)
            best_trial = trial_data["trial"]
            loss_map = {
                "MSE" : nn.MSELoss(),
                "Huber" : nn.HuberLoss(),
                "L1" : nn.L1Loss()
            }
            loss = loss_map[trial_data["loss"]]
            bs = trial_data["batch_size"]
            train_loader = DataLoader(train_val_set, shuffle=True, batch_size=int(bs))
            epochs = trial_data["epoch"]

            # Start the training
            for loop in range(1, repetitions+1):
                conf_folder_model = conf_folder / f"model_{loop}"
                conf_folder_model.mkdir(parents=True, exist_ok=True)
                torch.manual_seed(loop)
                print("Training and Testing Modell " + str(loop), flush=True)
                # Init + create a model based on the best trial
                mlp_model = create_optuna_mlp(best_trial)
                xavier_init(mlp_model)
                mlp_model.to(device)
                optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=trial_data["lr"], weight_decay=trial_data["weight_decay"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
                # Train on Train/Val Set
                for i in range(epochs):
                    _ = training(train_loader, mlp_model, loss, optimizer, device=device)

                    scheduler.step()

                intern_preds = []
                extern_preds = []
                # Amount of MC Models
                for i in range(50):
                    # Test on Intern and Extern
                    _, test_pred, test_target = testing(intern_test_loader, mlp_model, loss, device, mc_dropouts=True)
                    _, test_expred, test_extarget = testing(extern_test_loader, mlp_model, loss, device, mc_dropouts=True)
                    intern_preds.append(test_pred)
                    extern_preds.append(test_expred)



                intern_preds = np.array(intern_preds)
                extern_preds = np.array(extern_preds)
                intern_pred = np.mean(intern_preds, axis=0)
                extern_pred = np.mean(extern_preds, axis=0)
                intern_std = np.std(intern_preds, axis=0)
                extern_std = np.std(extern_preds, axis=0)                    


                intern_metrics = get_metrics(intern_pred, test_target, "Intern")
                extern_metrics = get_metrics(extern_pred, test_extarget, "Extern")
                
                if intern_metrics["MAE"] < min_mae_intern:
                    min_mae_intern = intern_metrics["MAE"]
                    best_intern_preds = intern_pred
                if extern_metrics["MAE"] < min_mae_extern:
                    min_mae_extern = extern_metrics["MAE"]
                    best_extern_preds = extern_pred
                    
                
                mlp_results["MLP_" + str(loop) + "_intern"] = intern_metrics
                mlp_results["MLP_" + str(loop) + "_extern"] = extern_metrics
                current_iter = {}
                current_iter["Intern"] = intern_metrics
                current_iter["Extern"] = extern_metrics
                current_df = pd.DataFrame(current_iter).T
                current_df.to_csv(f"{conf_folder}/model_{loop}/results.csv", index=False)
                
                test_df_intern_save = pd.read_csv(str(data_path / "Processed" / "Intern" / "Test" / (endpoint + "_chemprop_test.csv")))
                test_df_extern_save = pd.read_csv(str(data_path / "Processed" / "Extern" / "Test" / (endpoint + "_chemprop_test.csv")))
                test_df_intern_save["Preds"] = intern_pred
                test_df_extern_save["Preds"] = extern_pred
                test_df_intern_save[["smiles", "activity", "Preds"]].to_csv(f"{conf_folder}/model_{loop}/intern_preds.csv", index=False)
                test_df_extern_save[["smiles", "activity", "Preds"]].to_csv(f"{conf_folder}/model_{loop}/extern_preds.csv", index=False)
                
                # Train Preds
                train_loader = DataLoader(train_val_set, shuffle=False, batch_size=int(bs))
                train_preds = []
                # Amount of MC Models
                for i in range(50):
                    # Test on Intern and Extern
                    _, test_pred, test_target = testing(train_loader, mlp_model, loss, device, mc_dropouts=True)
                    train_preds.append(test_pred)
                    
                train_preds = np.array(train_preds)
                train_pred = np.mean(train_preds, axis=0)
                
                train_smiles = pd.read_csv(f"{data_path}/Processed/{dataset}/Train/{endpoint}_chemprop_train.csv")
                val_smiles = pd.read_csv(f"{data_path}/Processed/{dataset}/Val/{endpoint}_chemprop_val.csv")
                train_smiles = pd.concat((train_smiles, val_smiles), ignore_index=True)
                train_smiles["Preds"] = train_pred
                train_smiles.to_csv(f"{conf_folder}/model_{loop}/train_preds.csv", index=False)
                torch.save(mlp_model.state_dict(), f"{conf_folder}/model_{loop}/mlp_weights.pt")


            mlp_df = pd.DataFrame(mlp_results).T
            mlp_df.to_csv(str(data_path / "Evaluation" / "MLP" / dataset / (endpoint + "_results.csv")), index=False)


        # XGB schedule
        elif model == "XGB":

            xgb_results = {}
            trial_df = pd.read_csv(trial_path)
            trial_data = best_trial_xgb(trial_df)

            for loop in range(1, repetitions+1):
                conf_folder_model = conf_folder / f"model_{loop}"
                conf_folder_model.mkdir(parents=True, exist_ok=True)
                print("Training and Testing Modell " + str(loop), flush=True)
                xgb_model = xgboost.XGBRegressor(**trial_data)
                # Prevent Deterministic Behaviour
                xgb_model.random_state = loop 
                xgb_model.seed = loop
                X, y = train_val_df.drop("activity", axis=1), train_val_df.activity
                xgb_model.fit(X, y)
                X_test, y_test = test_df_intern.drop("activity", axis=1), test_df_intern.activity
                preds = xgb_model.predict(X_test)
                result_intern = get_metrics(preds, y_test, "Intern")

                X_test_extern, y_test_extern = test_df_extern.drop("activity", axis=1), test_df_extern.activity
                preds_extern = xgb_model.predict(X_test_extern)
                result_extern = get_metrics(preds_extern, y_test_extern, "Extern")
                
                if result_intern["MAE"] < min_mae_intern:
                    min_mae_intern = result_intern["MAE"]
                    best_intern_preds = preds
                if result_extern["MAE"] < min_mae_extern:
                    min_mae_extern = result_extern["MAE"]
                    best_extern_preds = preds_extern


                xgb_results["XGB_" + str(loop) + "_intern"] = result_intern
                xgb_results["XGB_" + str(loop) + "_extern"] = result_extern

                current_iter = {}
                current_iter["Intern"] = result_intern
                current_iter["Extern"] = result_extern
                current_df = pd.DataFrame(current_iter).T
                current_df.to_csv(f"{conf_folder}/model_{loop}/results.csv", index=False)
                
                test_df_intern_save = pd.read_csv(str(data_path / "Processed" / "Intern" / "Test" / (endpoint + "_chemprop_test.csv")))
                test_df_extern_save = pd.read_csv(str(data_path / "Processed" / "Extern" / "Test" / (endpoint + "_chemprop_test.csv")))
                test_df_intern_save["Preds"] = preds
                test_df_extern_save["Preds"] = preds_extern
                test_df_intern_save[["smiles", "activity", "Preds"]].to_csv(f"{conf_folder}/model_{loop}/intern_preds.csv", index=False)
                test_df_extern_save[["smiles", "activity", "Preds"]].to_csv(f"{conf_folder}/model_{loop}/extern_preds.csv", index=False)
                
                # Train Preds
                train_preds = xgb_model.predict(X)
                train_smiles = pd.read_csv(f"{data_path}/Processed/{dataset}/Train/{endpoint}_chemprop_train.csv")
                val_smiles = pd.read_csv(f"{data_path}/Processed/{dataset}/Val/{endpoint}_chemprop_val.csv")
                train_smiles = pd.concat((train_smiles, val_smiles), ignore_index=True)
                train_smiles["Preds"] = train_preds
                train_smiles.to_csv(f"{conf_folder}/model_{loop}/train_preds.csv", index=False)
                
            xgb_df = pd.DataFrame(xgb_results).T
            xgb_df.to_csv(str(data_path / "Evaluation" / "XGB" / dataset / (endpoint + "_results.csv")), index=False)

        # Chemprop schedule
        elif model == "Chemprop":
            chemprop_results = {}
            trial_data = best_trial_chemprop(trial_df)
            bs = int(trial_data["batch_size"])
            if dataset == "MT":
                test_loader = chemprop_loader_from_df(test_df, shuffle=False, bs=len(test_df), n_tasks=2, target_columns=["activity_intern", "activity_extern"])
                train_loader = chemprop_loader_from_df(train_val_df, shuffle=True, bs=bs, n_tasks=2, target_columns=["activity_intern", "activity_extern"])
                tasks = 2
                train_loader_testing = chemprop_loader_from_df(train_val_df, shuffle=False, bs=bs, n_tasks=tasks, target_columns=["activity_intern", "activity_extern"])
            else:
                intern_test_loader = chemprop_loader_from_df(test_df_intern, shuffle=False, bs=len(test_df_intern))
                extern_test_loader = chemprop_loader_from_df(test_df_extern, shuffle=False, bs=len(test_df_extern))
                tasks = 1
                
                train_loader = chemprop_loader_from_df(train_val_df, shuffle=True, bs=bs)
                train_loader_testing = chemprop_loader_from_df(train_val_df, shuffle=False, bs=bs, n_tasks=tasks)
            for loop in range(1, repetitions+1):
                conf_folder_model = conf_folder / f"model_{loop}"
                conf_folder_model.mkdir(parents=True, exist_ok=True)
                torch.manual_seed(loop)
                print("Training and Testing Modell " + str(loop))
                chemprop = create_optuna_chemprop(trial_data, 
                                                  n_tasks=tasks)

                trainer = pl.Trainer(
                        logger=False,
                        accelerator="gpu",
                        devices=1,
                        max_epochs=int(trial_data["max_epoch"]))

                trainer.fit(chemprop, train_loader)
                if dataset != "MT":
                    intern_pred = torch.concat(trainer.predict(chemprop, intern_test_loader)).numpy()
                    extern_pred = torch.concat(trainer.predict(chemprop, extern_test_loader)).numpy()
                    targets_intern = test_df_intern.activity.values
                    targets_extern = test_df_extern.activity.values
                    intern_metrics = get_metrics(intern_pred.reshape(-1), targets_intern, "Intern")
                    extern_metrics = get_metrics(extern_pred.reshape(-1), targets_extern, "Extern")
                    current_iter = {}
                    current_iter["Intern"] = intern_metrics
                    current_iter["Extern"] = extern_metrics
                    current_df = pd.DataFrame(current_iter).T
                    current_df.to_csv(f"{conf_folder}/model_{loop}/results.csv", index=False)

                    test_df_intern_save = pd.read_csv(str(data_path / "Processed" / "Intern" / "Test" / (endpoint + "_chemprop_test.csv")))
                    test_df_extern_save = pd.read_csv(str(data_path / "Processed" / "Extern" / "Test" / (endpoint + "_chemprop_test.csv")))
                    test_df_intern_save["Preds"] = intern_pred.reshape(-1)
                    test_df_extern_save["Preds"] = extern_pred.reshape(-1)
                    test_df_intern_save[["smiles", "activity", "Preds"]].to_csv(f"{conf_folder}/model_{loop}/intern_preds.csv", index=False)
                    test_df_extern_save[["smiles", "activity", "Preds"]].to_csv(f"{conf_folder}/model_{loop}/extern_preds.csv", index=False)
                    # Train Preds
                    train_preds = torch.concat(trainer.predict(chemprop, train_loader_testing)).numpy().reshape(-1)
                    train_smiles = pd.read_csv(f"{data_path}/Processed/{dataset}/Train/{endpoint}_chemprop_train.csv")
                    val_smiles = pd.read_csv(f"{data_path}/Processed/{dataset}/Val/{endpoint}_chemprop_val.csv")
                    train_smiles = pd.concat((train_smiles, val_smiles), ignore_index=True)
                    train_smiles["Preds"] = train_preds
                    train_smiles.to_csv(f"{conf_folder}/model_{loop}/train_preds.csv", index=False)
                    
                else:
                    preds = pd.DataFrame(torch.concat(trainer.predict(chemprop, test_loader)).numpy())
                    targets_intern = test_df.activity_intern
                    targets_extern = test_df.activity_extern
                    metrics = []
                    intern_extern_preds = []
                    for i, target in enumerate([targets_intern, targets_extern]):
                        mask = ~target.isna()
                        valid_preds = preds[i][mask].dropna()
                        metric = get_metrics(valid_preds, target.dropna(), "Intern" if i == 0 else "Extern")
                        metrics.append(metric)
                        intern_extern_preds.append(valid_preds)
                        df = pd.DataFrame({"activity" : target[mask],
                                           "Preds" : valid_preds})
                        label = "intern" if i == 0 else "extern"
                        df.to_csv(f"{conf_folder}/model_{loop}/{label}_preds.csv", index=False)
                    intern_metrics, extern_metrics = metrics
                    intern_pred, extern_pred = intern_extern_preds
                    current_iter = {}
                    current_iter["Intern"] = intern_metrics
                    current_iter["Extern"] = extern_metrics
                    current_df = pd.DataFrame(current_iter).T
                    current_df.to_csv(f"{conf_folder}/model_{loop}/results.csv", index=False)
                    # Train Preds
                    train_preds = torch.concat(trainer.predict(chemprop, train_loader_testing)).numpy()
                    targets_intern = train_val_df.activity_intern
                    targets_extern = train_val_df.activity_extern
                    intern_extern_preds = []
                    for i, target in enumerate([targets_intern, targets_extern]):
                        mask = ~target.isna()
                        valid_preds = train_preds[:, i][mask]
                        label = "intern" if i == 0 else "extern"
                        df = pd.DataFrame({"activity" : target[mask],
                                            "Preds" : valid_preds})
                        df.to_csv(f"{conf_folder}/model_{loop}/{label}_train_preds.csv", index=False)
                    
                if intern_metrics["MAE"] < min_mae_intern:
                    min_mae_intern = intern_metrics["MAE"]
                    best_intern_preds = intern_pred
                if extern_metrics["MAE"] < min_mae_extern:
                    min_mae_extern = extern_metrics["MAE"]
                    best_extern_preds = extern_pred
                
                chemprop_results["Chemprop_" + str(loop) + "_intern"] = intern_metrics
                chemprop_results["Chemprop_" + str(loop) + "_extern"] = extern_metrics

                torch.save(chemprop.state_dict(), f"{conf_folder}/model_{loop}/chemprop_weights.pt")

            chemprop_df = pd.DataFrame(chemprop_results).T
            chemprop_df.to_csv(str(data_path / "Evaluation" / "Chemprop" / dataset / (endpoint + "_results.csv")), index=False)

        if dataset != "MT":
            test_df_intern = pd.read_csv(str(data_path / "Processed" / "Intern" / "Test" / (endpoint + "_chemprop_test.csv")))
            test_df_extern = pd.read_csv(str(data_path / "Processed" / "Extern" / "Test" / (endpoint + "_chemprop_test.csv")))
            test_df_intern["Preds"] = best_intern_preds
            test_df_extern["Preds"] = best_extern_preds
            test_df_intern[["smiles", "activity", "Preds"]].to_csv(str(data_path / "Evaluation" / model / dataset / (endpoint + "_internpreds.csv")), index=False)
            test_df_extern[["smiles", "activity", "Preds"]].to_csv(str(data_path / "Evaluation" / model / dataset / (endpoint + "_externpreds.csv")), index=False)
        else:
            test_df = pd.read_csv(str(data_path / "Processed" / "Multitask" / "Test" / (endpoint + "_chemprop_test.csv")))
            test_df["Intern Preds"] = best_intern_preds
            test_df["Extern Preds"] = best_extern_preds
            test_df.to_csv(str(data_path / "Evaluation" / "Chemprop" / "MT" / (endpoint + "_preds.csv")), index=False)
            