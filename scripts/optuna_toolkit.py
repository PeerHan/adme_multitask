#!/usr/bin/env python3

import torch
from torch import nn
from torch_toolkit import *
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from numpy import mean, std
import pandas as pd
import numpy as np
import chemprop
from lightning import pytorch as pl

def chemprop_objective(
        trial : "Trial",
        train_df : "DataFrame",
        val_df : "DataFrame",
        target_columns : str | list = "activity",
        n_tasks : int = 1,
        log_path : str = ""
    
) -> float:
    """
    Function to build a MPNN network (chemprop) based on selected parameter which optuna will find.
    Furthermore the network will be trained with different parameter to find the best HP for the model. A simple logging is provided to track the amount of finished trials.
    
    Parameter :
        - trial : Optuna trial object
        - train_df : Dataframe with train data which should be split in train and val
        - val_df : Dataframe for hp validation
        - target_columns : name of the target(s) - can be multiple
        - n_tasks : specifiy for multitask
        - log_path : Folder for log files
    
    The following hyper parameter will be optimized:
        - depth for MP layer
        - hidden dim of MP layer
        - activation function for MP layer
        - dropout prob. for MP layer
        - aggregation method
        - amount of layers for FFN
        - droput prob for FFN
        - activation function for FFN
        - Use batchnorm after MP 
        - warmup epochs
        - max epochs
        - initial lr
        - batchsize
    
    
    """
    train_smiles = train_df.loc[:, "smiles"].values
    train_targets = train_df.loc[:, target_columns].values
    if n_tasks == 1:
        train_targets = train_targets.reshape(-1 , 1)
    train_data = [chemprop.data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(train_smiles, train_targets)]
    featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = chemprop.data.MoleculeDataset(train_data, featurizer)
    #scaler = train_dset.normalize_targets()
    
    val_smiles = val_df.loc[:, "smiles"].values
    val_targets = val_df.loc[:, target_columns].values
    if n_tasks == 1:
        val_targets = val_targets.reshape(-1, 1)
    val_data = [chemprop.data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(val_smiles, val_targets)]
    val_dset = chemprop.data.MoleculeDataset(val_data, featurizer)
    #val_dset = val_dset.normalize_targets(scaler)
    batch_size = trial.suggest_int("batch_size", 16, 512, step=16)

    train_loader = chemprop.data.build_dataloader(train_dset, num_workers=5, shuffle=True, batch_size=batch_size)
    val_loader = chemprop.data.build_dataloader(val_dset, num_workers=5, shuffle=False, batch_size=batch_size)

    
    # Start Optuna
    depth = trial.suggest_int("depth", 2, 5)
    hidden_dim_bond = trial.suggest_int("hidden_bond", 100, 800, 100)
    activation_bond = trial.suggest_categorical("activation_bond", choices=["relu", "leakyrelu", "prelu", "elu"])
    dropout = trial.suggest_float("dropout_bond", 0., 0.4, step=0.05)
    
    mp = chemprop.nn.BondMessagePassing(d_h=hidden_dim_bond, activation=activation_bond, dropout=dropout, depth=depth)
    aggregation = trial.suggest_categorical("aggregation", choices=["sum", "mean", "norm"])
    agg_map = {"sum" : chemprop.nn.SumAggregation(), "mean" : chemprop.nn.MeanAggregation(), "norm" : chemprop.nn.NormAggregation()}
    agg = agg_map[aggregation]
    
    n_layers = trial.suggest_int("layer", 1, 6)
    activation_ffn = trial.suggest_categorical("activation_ffn", choices=["relu", "leakyrelu", "prelu", "elu"])

    ffn_hidden_dim = trial.suggest_int("hidden_ffn", 100, 1500, 100)
    ffn_dropout = trial.suggest_float("dropout_ffn", 0, 0.4, step=0.05)
    ffn = chemprop.nn.RegressionFFN(output_transform=None, input_dim=hidden_dim_bond, hidden_dim=ffn_hidden_dim, n_layers=n_layers, activation=activation_ffn, n_tasks=n_tasks, dropout=ffn_dropout)

    
    bn = trial.suggest_categorical("BN", choices=["True", "False"])
    batch_norm = bn == "True"
    metric_list = [chemprop.nn.metrics.MAEMetric(), chemprop.nn.metrics.RMSEMetric()]
    warmup_epochs = trial.suggest_int("warmup_epochs", 2, 15)
    max_epoch = trial.suggest_int("max_epoch", 10, 150, 10)
    init_lr = trial.suggest_float("init_lr", 1e-4, 1e-1, log=True)
    
    model = chemprop.models.MPNN(mp, agg, ffn, batch_norm, metric_list, warmup_epochs=warmup_epochs, init_lr=init_lr)
    
    early_stopping = EarlyStopping('val_loss', patience=20, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    trainer = pl.Trainer(
            logger=False,
            accelerator="gpu",
            devices=1,
            max_epochs=max_epoch,
            callbacks=[early_stopping, checkpoint_callback])
    
    trainer.fit(model, train_loader, val_loader)
    model = model.load_from_checkpoint(checkpoint_callback.best_model_path)

    
    out = trainer.predict(model, val_loader)
    preds = pd.DataFrame(torch.concat(out).numpy())
    targets = val_df[target_columns]
    res = trainer.test(model, val_loader)
    mae = res[0]["batch_averaged_test/mae"]
    r2 = 0
        
    if n_tasks == 1:
        r2 = pd.concat((targets, preds), axis=1).corr().iloc[0, 1] ** 2
    # Multitask
    else:
        mask = ~targets.isna()
        valid_preds = preds[mask]
        r2s = []
        for i, col in enumerate(target_columns):
            mask = ~targets[col].isna()
            r2 = np.corrcoef(preds[i][mask].dropna(), targets[col].dropna()) ** 2
            r2s.append(r2)
        r2 = mean(r2s)

    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(r2) + " - MAE : " + str(mae) + "\n")
        
    return r2, mae


def create_optuna_chemprop(
        best_model : "Series",
        n_tasks : int = 1,
    ) -> "Model":
    """
    Function to create a pytorch chemprop model based on the best HP-Search with optuna.
    The returned model only consists of the following HP:
        - depth for MP layer
        - hidden dim of MP layer
        - activation function for MP layer
        - dropout prob. for MP layer
        - aggregation method
        - amount of layers for FFN
        - droput prob for FFN
        - activation function for FFN
        - Use batchnorm after MP 
        - warmup epochs
        - max epochs
        - initial lr
    
    Parameters:
        - best_model : Pandas series object of the best optuna trial
        - n_tasks : Single or Multitask
    """
    
    depth = best_model["depth"]
    hidden_dim_bond = best_model["hidden_bond"]
    activation_bond = best_model["activation_bond"]
    dropout = best_model["dropout_bond"]
    
    mp = chemprop.nn.BondMessagePassing(d_h=hidden_dim_bond, activation=activation_bond, dropout=dropout, depth=depth)
    aggregation = best_model["aggregation"]
    agg_map = {"sum" : chemprop.nn.SumAggregation(), "mean" : chemprop.nn.MeanAggregation(), "norm" : chemprop.nn.NormAggregation()}
    agg = agg_map[aggregation]
    
    n_layers = best_model["layer"]
    activation_ffn = best_model["activation_ffn"]

    ffn_hidden_dim = best_model["hidden_ffn"]
    ffn_dropout = best_model["dropout_ffn"]
    ffn = chemprop.nn.RegressionFFN(output_transform=None, input_dim=hidden_dim_bond, hidden_dim=ffn_hidden_dim, n_layers=n_layers, activation=activation_ffn, n_tasks=n_tasks, dropout=ffn_dropout)

    
    bn = best_model["BN"]
    batch_norm = bn == "True"
    metric_list = [chemprop.nn.metrics.MAEMetric(), chemprop.nn.metrics.RMSEMetric()]
    warmup_epochs = best_model["warmup_epochs"]
    max_epoch = best_model["max_epoch"]
    init_lr = best_model["init_lr"]
    
    model = chemprop.models.MPNN(mp, agg, ffn, batch_norm, metric_list, warmup_epochs=warmup_epochs, init_lr=init_lr)
    

    return model

def define_regressor(
        trial : "Trial", 
        input_feats : int, 
        output : int = 1,
    ) -> "Model":
    """
    Function to build a regression model based on optuna.
    
    Parameters:
        - trial : Optuna trial object
        - input_feats : Number of features as input features for the network
        - output : Number of regresion tasks
    
    Hyparameter:
        - Amount of layers (1-10)
        - Activation function (ReLU, Mish, SiLU, leaky ReLU)
        - Hidden Units (10 - 500)
        - Dropout prob (0 - 0.4)
    """
    # HP 1 : How many Layers?
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []
    # HP 2 : Which activation
    activation = trial.suggest_categorical("Activation", choices=["RELU", "Mish", "Swish", "Leaky"])
    
    activation_func = {
        "RELU" : nn.ReLU(),
        "Mish" : nn.Mish(),
        "Swish" : nn.SiLU(),
        "Leaky" : nn.LeakyReLU(),
    }
    for i in range(n_layers):
        # HP 3 : Amount of Hidden units
        output_feats = trial.suggest_int("neurons_" + str(i), 100, 1000, step=100)
        layers.append(nn.Linear(input_feats, output_feats))
        layers.append(activation_func[activation])
        # HP 4 : Dropout Probability
        #if activation != "SELU":
         #   layers.append(nn.BatchNorm1d(output_feats))
        if i < n_layers - 1:
            p = trial.suggest_float("dropout_" + str(i), 0., 0.4, step=0.05)
            layers.append(nn.Dropout(p))
        input_feats = output_feats
    
    layers.append(nn.Linear(input_feats, output))
    
    return nn.Sequential(*layers), activation

def mlp_objective(
        trial : "Trial", 
        train_df : "DataFrame", 
        val_df : "DataFrame",
        target : str = "activity",
        device : str = "cuda", 
        input_feats : int = 1340,
        log_path : str = "mlp_logs.txt",
    ) -> float:
    """
    Function to build a network based on selected parameter which optuna will find.
    Furthermore the network will be trained with different parameter to find the best HP for the model. A simple logging is provided to track the amount of finished trials.
    
    Parameter :
        - trial : Optuna trial object
        - train_df : Dataframe with train data which should be split in train and val
        - val_df : Dataframe for hp validation
        - target : name of the target
        - device : Should be always cuda if a gpu is available
        - input_feats : Input feats for the network
        - log_path : Folder for log files
    
    The following hyper parameter will be optimized:
        - Layer
        - Activation Function
        - Hidden units per layer
        - Dropout probability
        - Batchsize
        - LR Scheduler patience
        - Model initialization
        - Learning rate
        - Weight decay
    """
    
    # HP 5 : Batchsize
    batch_size = trial.suggest_int("batch_size", 16, 512, step=16)
    patience = 20
    delta = 0.
    # HP 6 : LR Scheduler patience
    lr_pat = trial.suggest_int("lr_patience", 5, 10, step=1)
    epochs = 1000
    
    # Model Section
    model, activation = define_regressor(trial, input_feats=1340, output=1)
    
    # HP 7: Learning Rate
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # HP 8 : Weight Decay
    wd = trial.suggest_float("weight_decay", 1e-9, 1e-5, log=True)
    loss = trial.suggest_categorical("loss", choices=["MSE", "Huber", "L1"])
    
    loss_map = {
        "MSE" : nn.MSELoss(),
        "Huber" : nn.HuberLoss(),
        "L1" : nn.L1Loss()
    }
    
    loss_f = loss_map[loss]
    
    r2 = None
    mae = None
    last_epoch = None

    train_loader, val_loader = create_dataloader(
        {"Train" : train_df,
         "Test" : val_df
        },
        batch_size=batch_size,
        target=target
    ).values()

    model = model.to(device)
    xavier_init(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=lr_pat)
    early_stopper = EarlyStopper(patience=patience, min_delta=delta)
    print("Starting Training", flush=True)
    for epoch in range(epochs):
        t_loss = training(train_loader, model, loss_f, optimizer)
        v_loss = validation(val_loader, model, loss_f, optimizer)

        scheduler.step(v_loss)

        if early_stopper.early_stop(v_loss.item()):
            break
    last_epoch = epoch


    _, test_pred, test_target = testing(val_loader, model, loss_f)

    r2 = np.corrcoef(test_target, test_pred)[0, 1] ** 2
    mae = mean_absolute_error(test_target, test_pred)

    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(r2) + " - MAE : " + str(mae) + " - Epochs " + str(last_epoch) + "\n")
        
    return r2, mae

def rf_objective(
        trial : "Trial", 
        train_df : "DataFrame", 
        val_df : "DataFrame",
        target : str = "activity",
        log_path : str = ".",
    ) -> (float, float, float):
    """
    Function to find the best hyperparameter for a RandomForest model based on the validation data.
    
    Parameter :
        - trial : Optuna trial object
        - train_df : Dataframe with train data
        - val_df : Dataframe with validation data which will be used to find the HP
        - target : name of the target
        - log_path : Folder for log files
    
    The following hyper parameter will be optimized:
        - criterion
        - max_depth
        - min_samples_split
        - min_samples_leaf
        - max_features
        - max_leaf_nodes
        - max_samples
    """

    X_train, y_train = train_df.drop(target, axis=1), train_df[target]
    X_val, y_val = val_df.drop(target, axis=1), val_df[target]
    
    n_estimators = trial.suggest_int(name="n_estimators", low=100, high=1000, step=100)
    criterion = trial.suggest_categorical(name="criterion", choices=["squared_error", "absolute_error", "friedman_mse", "poisson"])
    max_features = trial.suggest_categorical(name="max_features", choices=[None, 'sqrt', "log"]) 
    max_depth = trial.suggest_int(name="max_depth", low=10, high=510, step=20)
    min_samples_split = trial.suggest_int(name="min_samples_split", low=2, high=50, step=2)
    min_samples_leaf = trial.suggest_int(name="min_samples_leaf", low=1, high=4, step=1)
    
    params = {
        "n_estimators": n_estimators,
        "criterion" : criterion,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "n_jobs" : 15
    }
    
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    
    test_pred = rf.predict(X_val)
    
    r2 = r2_score(y_val, test_pred)
    mse = mean_squared_error(y_val, test_pred)
    mae = mean_absolute_error(y_val, test_pred)
        
    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(r2) + " - MAE : " + str(mae) + " - MSE : " + str(mse) + "\n")
        
    return r2, mse, mae

def elastic_objective(
        trial : "Trial", 
        train_df : "DataFrame", 
        val_df : "DataFrame",
        target : str = "activity",
        log_path : str = ".",
    ) -> (float, float, float):
    """
    Function to find the best hyperparameter for an ElasticNet model based on the validation data.
    
    Parameter :
        - trial : Optuna trial object
        - train_df : Dataframe with train data
        - val_df : Dataframe with validation data which will be used to find the HP
        - target : name of the target
        - log_path : Folder for log files
    
    The following hyper parameter will be optimized:
        - alpha
        - l1 ratio
        - max_iter
    """

    X_train, y_train = train_df.drop(target, axis=1), train_df[target]
    X_val, y_val = val_df.drop(target, axis=1), val_df[target]
    
    alpha = trial.suggest_float(name="alpha", low=0.1, high=5., step=0.1)
    l1_ratio = trial.suggest_float(name="l1_ratio", low=0., high=1., step=0.05)
    max_iter = trial.suggest_int(name="max_iter", low=1000, high=5000, step=100) 
    
    params = {
        "alpha" : alpha,
        "l1_ratio" : l1_ratio,
        "max_iter" : max_iter,
        "n_jobs" : 15,
    }
    
    en = ElasticNet(**params)
    en.fit(X_train, y_train)
    
    test_pred = en.predict(X_val)
    
    r2 = r2_score(y_val, test_pred)
    mse = mean_squared_error(y_val, test_pred)
    mae = mean_absolute_error(y_val, test_pred)
        
    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(r2) + " - MAE : " + str(mae) + " - MSE : " + str(mse) + "\n")
        
    return r2, mse, mae

def xgb_objective(
        trial : "Trial", 
        train_df : "DataFrame", 
        val_df : "DataFrame",
        target : str = "activity",
        log_path : str = "xgb_log.txt",
    ) -> (float, float, float):
    """
    Function to find the best hyperparameter for an XGBoost model based on the validation data.
    
    Parameter :
        - trial : Optuna trial object
        - train_df : Dataframe with train data
        - val_df : Dataframe with val data
        - target : name of the target
        - log_path : Folder for log files 
    
    The following hyper parameter will be optimized:
        - max_depth
        - learning_rate
        - n_estimators
        - min_child_weight
        - gamma
        - subsample
        - colsample_bytree
        - reg_alpha
        - reg_lambda
    """
    
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 75),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 0., 3.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 3.0),
        'eta' : trial.suggest_float('eta', 0., 1.0),
        'n_jobs' : 30
    }
    
    mae = None
    r2 = None

    X_train, y_train = train_df.drop(target, axis=1), train_df[target]
    X_val, y_val = val_df.drop(target, axis=1), val_df[target]

    xgb = xgboost.XGBRegressor(**param)
    xgb.fit(X_train, y_train)

    test_pred = xgb.predict(X_val)

    r2 = np.corrcoef(y_val, test_pred)[0, 1] ** 2
    mae = mean_absolute_error(y_val, test_pred)
    
    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(r2) + " - MAE : " + str(mae) + "\n")
        
    return r2, mae


def create_optuna_mlp(
        best_model : "Series",
        outputs : int = 1,
        input_feats : int = 1340,
    ) -> "Model":
    """
    Function to create a pytorch model based on the best HP-Search with optuna.
    The returned model only consists of the following HP:
        - Layers
        - Neurons per layer
        - Dropout probability per layer
        - Activation function
    
    Parameters:
        - best_model : Pandas series object of the best optuna trial
        - input_feats : Amount of input features for the first layer
    """
    
    layers = []
    N = best_model.params_n_layers
    activation = best_model.params_Activation
    input_feats = input_feats
    
    activation_func = {
        "RELU" : nn.ReLU(),
        "Mish" : nn.Mish(),
        "Swish" : nn.SiLU(),
        "Leaky" : nn.LeakyReLU(),
    }
    
    for i in range(N):
        output_feats = best_model["params_neurons_" + str(i)]
        dropout_p = 0 if (i == N-1) else best_model["params_dropout_" + str(i)]
        input_feats = int(input_feats)
        output_feats = int(output_feats)
        layers.append(nn.Linear(input_feats, output_feats))
        layers.append(activation_func[activation])
        layers.append(nn.Dropout(dropout_p))
        input_feats = output_feats
    
    # delete last dropout
    del layers[-1]
    layers.append(nn.Linear(input_feats, outputs))
    
    model = nn.Sequential(*layers)

    return model
        
        
def best_trial_mlp(
        study_df : "DataFrame",
        log_path : str,
    ) -> dict:
    """
    Function to get all HPs of the best optuna trial (MLP) and the corresponding pytorch model.
    All objects are stored in a dictionary with the following key-value pairs:
        - batch_size : Batch size for training
        - init : Weight initializing for the network
        - lr : Initial learning rate
        - lr_patience : Patience for the learning rate scheduler
        - weight_decay : Weight decay regularization value
        - epoch : Amount of epochs
        - trial : Pandas Series of the best model
        
    Parameters:
        - study_df : Dataframe from an optuna study
    """
    important_params = ["params_batch_size", "params_lr",
                        "params_lr_patience", "params_weight_decay", "params_loss"]
        
    for q in [0.01, 0.025, 0.05, 0.075, 0.1]:
        subset = study_df[(study_df.R2 >= study_df.R2.quantile(1-q)) & (study_df.MAE <= study_df.MAE.quantile(q))]
        if len(subset) != 0:
            break
    if len(subset) == 0:
        subset = study_df                  
    best_model = subset.sort_values("R2", ascending=False).reset_index(drop=True).iloc[0, :]
    
    trial_data = {param.replace("params_", "") : best_model[param] for param in important_params}
    
    trial_data["trial"] = pd.Series(best_model)
    
    trial_number = trial_data["trial"].number
    

    with open(log_path, "r") as file:
        for line in file.readlines():
            epoch = int(line.strip().split(" ")[-1])
            trial_num = int(line.strip().split(" ")[3])
            if trial_number == trial_num:
                trial_data["epoch"] = epoch
                break
    
    return trial_data

def best_trial_xgb(
        study_df : "DataFrame",
        n_jobs=30
    ) -> dict:
    """
    Function to get all Hps for the best optuna trial (XGB) and the corresponding xgb-model.
    All parameters are stored in a dictionary with the following key-parameter pairs:
        - max_depth 
        - learning_rate
        - n_estimators
        - min_child_weight
        - gamma
        - subsample
        - colsample_bytree
        - reg_alpha
        - reg_lambda
        - n_jobs
    """
    
    important_params = ["params_max_depth", "params_learning_rate", "params_n_estimators",
                        "params_min_child_weight", "params_gamma", "params_subsample",
                        "params_colsample_bytree", "params_reg_alpha", "params_reg_lambda"]
    
    for q in [0.01, 0.025, 0.05, 0.075, 0.1]:
        subset = study_df[(study_df.R2 >= study_df.R2.quantile(1-q)) & (study_df.MAE <= study_df.MAE.quantile(q))]
        if len(subset) != 0:
            break
    if len(subset) == 0:
        subset = study_df      
    best_model = subset.sort_values("R2", ascending=False).reset_index(drop=True).iloc[0, :]
        
    trial_data = {param.replace("params_", "") : best_model[param] for param in important_params}
    
    trial_data["n_jobs"] = n_jobs
    
    return trial_data

def best_trial_chemprop(
        study_df : "DataFrame",
    ) -> dict:
    """
    Function to get all HPs of the best optuna trial (chemprop GNN) and the corresponding pytorch model.
    All objects are stored in a dictionary with key value pairs to rebuild the best model.
        
    Parameters:
        - study_df : Dataframe from an optuna study
    """
    important_params = ['params_BN', 'params_activation_bond',
       'params_activation_ffn', 'params_aggregation', 'params_batch_size',
       'params_depth', 'params_dropout_bond', 'params_dropout_ffn',
       'params_hidden_bond', 'params_hidden_ffn', 'params_init_lr',
       'params_layer', 'params_max_epoch', 'params_warmup_epochs']
    for q in [0.01, 0.025, 0.05, 0.075, 0.1]:
        subset = study_df[(study_df.R2 >= study_df.R2.quantile(1-q)) & (study_df.MAE <= study_df.MAE.quantile(q))]
        if len(subset) != 0:
            break

    if len(subset) == 0:
        subset = study_df    
    best_model = subset.sort_values("R2", ascending=False).reset_index(drop=True).iloc[0, :]
    
    trial_data = {param.replace("params_", "") : best_model[param] for param in important_params}
    
    trial_data["trial"] = pd.Series(best_model)
    
    return trial_data