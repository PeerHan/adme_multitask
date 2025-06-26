#!/usr/bin/env python3

from torch.utils.data import (
        DataLoader, 
        Dataset,
    )

from torch import nn
import torch
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from random import random
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.regression.linear_model import RegressionResults
from sklearn.preprocessing import RobustScaler, StandardScaler
import pandas as pd
from sklearn.model_selection import KFold
import plotly.io as pio
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassCohenKappa
from scipy.stats import spearmanr
import chemprop

# ---------------------------- Chemprop ------------------------

def chemprop_loader_from_df(
        df : "DataFrame",
        target_columns : str | list = "activity",
        n_tasks : int = 1,
        bs : int = 16,
        shuffle : bool = False
        ) -> "DataLoader":
    """
    Create a Dataloader for chemprop pytorch lightning models.
    Parameters:
        - df : Dataframe for the loader
        - target_columns : can be one col for single task or a list with tasks
        - n_tasks : specify if single or multitask
        - bs : batch size
        - shuffle : toggle for test/val and train loader
    """
    smiles = df.loc[:, "smiles"].values
    targets = df.loc[:, target_columns].values
    if n_tasks == 1:
        targets = targets.reshape(-1, 1)
    data = [chemprop.data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, targets)]
    featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = chemprop.data.MoleculeDataset(data, featurizer)

    loader = chemprop.data.build_dataloader(dset, num_workers=5, shuffle=shuffle, batch_size=bs)
    return loader

# ---------------------------- Dataset and Dataloader ------------------------
class MixedDescriptorSet(Dataset):
    """
    Dataset for featurized smiles through a mix of 2D Descriptors (Morgan Fingerprint with 1024 feats) and 1D Descriptors (316)
    Parameters:
        - df : Dataframe containing features, target and ID
        - target : Target name - default name is 'activity'
        - id_col : ID Column name - default name is 'ID'
        - scale : Bool for scaling the 1D Descriptor values with a Standard Scaler
    """
    
    def __init__(
        self, 
        df : "DataFrame", 
        target : str = "activity", 
        id_col : str =" ID", 
        scale : bool = True,
    ) -> None:
        if id_col in df.columns:
            df = df.drop(id_col, axis=1)
        self.y = df[target]
        self.X = df.drop([target], axis=1).astype(float)
        if scale:
            scaler = StandardScaler()
            scaled_1d = scaler.fit_transform(self.X.filter(like="rdMD_"))
            self.X = np.hstack((self.X.filter(like="fcfp4_").values, scaled_1d))
    
    def __len__(
        self,
    ) -> int:
        return len(self.y)
    
    def __getitem__(
        self, 
        index : int,
    ) -> "(Tensor, Tensor)":
        x = self.X[index, :]
        y = self.y[index]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


def create_dataloader(
        df_map : dict, 
        batch_size : int = 32, 
        target : str = "activity", 
        shuffle : bool = True, 
        drop_last : bool = False, 
        dataset : Dataset = MixedDescriptorSet, 
        scale : bool = True,
    ) -> "dict(str, Dataloader)":
    """
    Function to create a dictionary of dataloader from a dictionary of dataframes.
    The keys correspond to the data set : 'Train', 'Val' and 'Test'.
    dict -> dict
    
    Parameters : 
        - df_map : Dictionary with the keys 'Train', 'Val' and 'Test
        - batch_size : Batch size for the dataloader - Default value is 32
        - target : Name for the target column - Default name is 'activity'
        - shuffle : Bool for the dataloader parameter, 
        this only affects the 'Train' and 'Val' dataloader. 
        - drop last : Bool for the dataloader parameter.
        - dataset : Dataset Class to specify which dataset should be used for the dataloader
        - scale : Bool to decide, if the 1D Descriptors should be scaled or not
   
    """
    loaders = {}
    
    for label, df in df_map.items():

        data_set = dataset(df, target=target, scale=scale)
        if label == "Test":
            loader = DataLoader(data_set, 
                                batch_size=1, 
                                shuffle=False, 
                                drop_last=False)
        else:
            loader = DataLoader(data_set,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last)
        loaders[label] = loader
    
    return loaders

class EarlyStopper:
    """
    Class to implement an early stop method as a regularization strategy
    
    Parameter:
        - patience : Value to decide when an early stop should come in
        - min_delta : Tolerance value when comparing a new value with the minimum value
    """
    
    def __init__(
        self,
        patience : int = 1,
        min_delta : float = 0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_value = float("inf")
    
    def early_stop(
        self,
        value : float,
    ) -> bool:
        if value < self.min_value:
            self.min_value = value
            self.counter = 0
        elif value >= (self.min_value + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
# ---------------------------- Regression ------------------------


def training(
        loader : "DataLoader", 
        model : "Model", 
        loss : "Loss", 
        optimizer : "Optim", 
        device : str = "cuda", 
    ) -> float:
    """
    Training function for a pyTorch regression model.
    The corresponding loss value will be returned.
    
    Parameters:
        - loader : Should be the train-loader
        - model : pyTorch Model for the regression task
        - loss : Loss function which should be used to train the model
        - optimizer : Optimizer which should be used to train the model
        - device : Should always specify the gpu
    """
    model.train()
    current_loss = 0.
    for x, y in loader:
        x = x.to(device)
        optimizer.zero_grad()
        out = model(x)
        out = out.to("cpu")
    
        y = y.unsqueeze(1)
        l = loss(out, y)
        current_loss += l
        l.backward()
        optimizer.step()
        
    return current_loss

def validation(
        loader : "DataLoader", 
        model : "Model", 
        loss : "Loss", 
        optimizer : "Optim", 
        device : str = "cuda", 
    ) -> float:
    """
    Validation function for a pyTorch regression model.
    The corresponding loss value will be returned.
    
    Parameters:
        - loader : Should be the validation-loader
        - model : pyTorch Model for the regression task
        - loss : Loss function which should be used to validate the model
        - optimizer : Optimizer which should be used to validate the model
        - device : Should always specify the gpu
    """
    model.eval()
    val_loss = 0
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        out = out.to("cpu")
        y = y.unsqueeze(1)
        l = loss(out, y)
        val_loss += l 
    return val_loss

@torch.no_grad()
def testing(
        loader : "DataLoader", 
        model : "Model", 
        loss : "Loss", 
        device : str = "cuda",
        mc_dropouts : bool = False
    ) -> [float, np.ndarray, np.ndarray]:
    """
    Test function for a pyTorch regression model.
    The corresponding test loss, predictions, ground truth values will be returned
    and can be used for further analysis purposes.
    Returns -> [Test_Loss, test_preds, test_truths]
    
    Parameters:
        - loader : Should be the train-loader
        - model : pyTorch Model for the regression task
        - loss : Loss function which should be used to train the model
        - optimizer : Optimizer which should be used to train the model
        - device : Should always specify the gpu
    """
    test_loss = 0
    test_target = np.empty((0))
    test_y_target = np.empty((0))
    if mc_dropouts:
        model.train()
    else:
        model.eval()
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        out = out.to("cpu")
        l = loss(out, torch.reshape(y, (len(y), 1)))
        test_loss += l.item()
        test_target = np.concatenate((test_target, out.detach().numpy()[:, 0]))
        test_y_target = np.concatenate((test_y_target, y.detach().numpy()))
    return test_loss, test_target, test_y_target

    
def train_and_validate(
        epochs : int, 
        model : "Model", 
        train_loader : "DataLoader", 
        val_loader : "DataLoader", 
        weight_path : str = None, 
        lr : float = 0.001, 
        wd : float = 5e-4, 
        loss : "Loss" = nn.HuberLoss(), 
        patience : int = 2, 
        delta : float = 0., 
        show_every : int = 2, 
        scheduler_pat : int = 3, 
        device : str = "cuda", 
        optim : "Optimizer" = torch.optim.Adam, 
    ):
    """
    Wrapping training and validation function to train/validate a new model.
    ReduceLROnPlateau is used as a lr scheduler which will decrease the learning rate, when the val loss is not improving.
    The function returns a list of Arrays : 
        [Train Loss per Epoch, Val Loss per Epoch, Train Prediction, Train Target, Val Prediction, Val Target]
    
    Parameters :
        - epochs : Number of epochs
        - model : Model Instance of a pyTorch Model
        - train_loader : DataLoader with training data
        - val_loader : DataLoader with validation data
        - weight_path : Path (incl. file name) to save the weights from the model with the lowest val loss
        - lr : Start learning rate
        - wd : weight decay
        - loss : Loss Function - Default function is HuberLoss
        - patience : Parameter for EarlyStop - Determines the Number of tolerated non improving epochs
        - delta : Parameter for EarlyStop - Determines the tolerance of EarlyStop value
        - show_every : Shows the print output every x epochs
        - scheduler_pat : Parameter for the Lr Scheduler - Determines how many non improving epochs (lower val los) are
        tolerated until the learning rate will be decreased
        - device : Should always be cude to enable a faster training on the gpu
        - optim : Choose optimizer
    """
    
    early_stopper = EarlyStopper(patience=patience, min_delta=delta)

    optimizer = optim(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=scheduler_pat)


    train_target = np.empty((0))
    train_y_target = np.empty((0))
    val_target = np.empty((0))
    val_y_target = np.empty((0))
    train_loss = np.empty(epochs)
    val_loss = np.empty(epochs)
    best_loss = np.inf

    for epoch in range(epochs):
        epoch_loss = training(train_loader, model, loss, optimizer, device=device)
        v_loss = validation(val_loader, model, loss, optimizer, device=device)
        if v_loss < best_loss and weight_path:
            best_loss = v_loss
            torch.save(model.state_dict(), weight_path)

        train_loss[epoch] = epoch_loss.detach().numpy()
        val_loss[epoch] = v_loss.detach().numpy()
        
        # Scheduler step
        if not optim:
            scheduler.step(v_loss)
            
        # print current train and val loss
        if epoch % show_every == 0:
            print(
                "Epoch: "
                + str(epoch)
                + ", Train loss: "
                + str(epoch_loss.item())
                + ", Val loss: "
                + str(v_loss.item())
            )
        
        if early_stopper.early_stop(v_loss.item()) or epoch == (epochs-1):
            print(f"Stop at epoch {epoch}")
            for x, y in train_loader:
                x = x.to(device)
                out = model(x)
                out = out.to("cpu")
                train_target = np.concatenate((train_target, out.detach().numpy()[:, 0]))
                train_y_target = np.concatenate((train_y_target, y.detach().numpy()))
            for x,y in val_loader:
                x = x.to(device)
                out = model(x)
                out = out.to("cpu")
                val_target = np.concatenate((val_target, out.detach().numpy()[:, 0]))
                val_y_target = np.concatenate((val_y_target, y.detach().numpy()))
            break
    # t loss, v loss, t pred, t y, val pred, v y
    return train_loss[:epoch+1], val_loss[:epoch+1], train_target, train_y_target, val_target, val_y_target
    
        
def calculate_metrics(target, pred, data_set):
    mse = mean_squared_error(target, pred)
    mae = mean_absolute_error(target, pred)
    r = np.corrcoef(target, pred)[0, 1]
    rmse = sqrt(mse)
    result = {
        f"{data_set}_MSE" : mse,
        f"{data_set}_MAE" : mae,
        f"{data_set}_R" : r,
        f"{data_set}_RMSE" : rmse,
        f"{data_set}_R2" : r ** 2
    }
    return result

def le_cun_init(model):
    for param in model.parameters():
        # biases zero
        if len(param.shape) == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
            
def xavier_init(model):
    for param in model.parameters():
        # biases zero
        if len(param.shape) == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_uniform_(param)
            
def kaiming_init(model):
    for param in model.parameters():
        # biases zero
        if len(param.shape) == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.kaiming_normal_(param, mode='fan_in')