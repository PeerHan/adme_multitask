import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from scipy import stats
import numpy as np
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from math import sqrt
from rdkit import Chem
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

sns.set_style("darkgrid")
sns.set(font_scale=1.25)

def get_metrics(
        preds : np.ndarray, 
        targets : np.ndarray,
        dataset : str,
    ) -> dict:
    """
    Function to calculate metrics based on the model prediction and the ground truth.
    The following metrics (sklearn) are calculated:
        - Mean absolute error
        - Mean squared error
        - Root mean squared error
        - Pearson R
        - R^2 
        - Spearman R
    
    Parameters:
        - preds : Prediction array
        - targets : Target array
        - dataset : Label for the test-set
    """
    
    results = {}
    results["MAE"] = mean_absolute_error(targets, preds)
    results["MSE"] = mean_squared_error(targets, preds)
    results["RMSE"] = sqrt(results["MSE"])
    results["R"] = np.corrcoef(targets, preds)[0, 1]
    results["R2"] = results["R"] ** 2
    results["SPR"] = spearmanr(targets, preds).statistic
    results["Testset"] = dataset
    
    return results

def model_df_on_set(
        endpoint : str, 
        data_path : str, 
        training_set : str,
    ) -> "DataFrame":
    """
    Function to create a combined dataframe per training set (Trained on intern or mixed (inex) data) on a specific endpoint.
    Parameters:
        - endpoint : Desired Endpoint (hPPB, rPPB, Sol, HLM, RLM)
        - data_path : Path to the set folder
        - training_set : Get the results from models which are trained either on Intern oder InEx (mixed) data
        - mlp_metric : Choose if the best MAE or R2 Trial should be used for the comparison
    """
    models = ["MLP", "XGB", "Chemprop"]
    model_dfs = [pd.read_csv(f"{data_path}/{model}/{training_set}/{endpoint}_results.csv") for model in models ]
    for model_df, model in zip(model_dfs, models):
        model_df["Model"] = model
    result_df = pd.concat(model_dfs, ignore_index=True)
    return result_df

def result_df_per_endpoint(
        endpoint : str, 
        data_path : str,
    ) -> "DataFrame":
    """
    Function to create a combined dataframe on a specific endpoint.
    The dataframe contains the evaluation results from 3 Models (MLP, XGB, Chemprop) on 30 Repetitions
    each on the extern- and the intern-testset. The dataframe also contains information about the used
    training set (either Intern data or mixed data (InEx)).
    The format is important for further visualization and testing functions.
    Parameters:
        - endpoint : Desired Endpoint (hPPB, rPPB, Sol, HLM, RLM)
        - data_path : Path to the set folder
        - mlp_metric : Choose if the best MAE or R2 Trial should be used for the comparison
    """
    result_dfs = []
    for training_set in ["Extern", "Intern", "InEx"]:
        result_df = model_df_on_set(endpoint, data_path, training_set)
        result_df["Training Set"] = training_set
        result_dfs.append(result_df)
    multi_df = pd.read_csv(f"{data_path}/Chemprop/Multitask/{endpoint}_results.csv")
    multi_df["Model"] = "Multitask"
    multi_df["Training Set"] = "InEx"
    result_dfs.append(multi_df)
    result_dfs = pd.concat(result_dfs, ignore_index=True)
    return result_dfs

def create_fp(
        df : "DataFrame", 
        nbits : int = 1024, 
        radius : int = 2,
    ) -> [list, list]:
    """
    Transforms smiles to MorganFingerprint.
    Returns fp and the smiles as lists.
    Parameters:
        - nbits : Amount of bits for the fp (default = 1024)
        - radius : Radius for the fp (default = 2)
    """
    smiles = df.smiles.tolist()
    ms = [Chem.MolFromSmiles(x) for x in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, nBits=nbits, radius=radius) for mol in ms]
    return fps, smiles

def calc_dice_sim(
        test_fps : list, 
        train_fps : list, 
        test_smiles : list, 
        train_smiles : list
    ) -> "DataFrame":
    """
    Creates a similarity dataframe based on the Sorensen Dice Coefficient.
    The dataframe has the following properties:
        - sm1 : smiles from the test df
        - sm2 : smiles from the train df
        - sim : similarity
    Parameters:
        - test_fps : test fingerprints
        - train_fps : train fingerprints
        - test_smiles : test smiles
        - train_smiles : train smiles
    """
    qu, ta, sim = [], [], []
    for n in range(len(test_fps)):
        s = DataStructs.BulkDiceSimilarity(test_fps[n], train_fps[:]) 
        # collect the SMILES and values
        for m in range(len(s)):
            qu.append(test_smiles[n])
            ta.append(train_smiles[:][m])
            sim.append(s[m])
    sim_df = pd.DataFrame({"sm1" : qu, "sm2" : ta, "sim" : sim})
    return sim_df

def get_sim_df(
        test_df : "DataFrame", 
        train_df : "DataFrame", 
        nbits : int = 1024, 
        radius : int = 2
    ) -> "DataFrame":
    """
    Wrapper function to create a smilarity dataframe based on the sorensen dice coefficient 
    based on a train and test dataframe (smile_col = smiles).
    Parameters:
        - test_df : Dataframe with test smiles
        - train_df : Dataframe with train smiles
        - nbits and radius : Parameters for the morgan fingerprint
    """
    train_fps, train_smiles = create_fp(train_df, nbits, radius)
    test_fps, test_smiles = create_fp(test_df, nbits, radius)
    sim_df = calc_dice_sim(test_fps, train_fps, test_smiles, train_smiles)
    sim_df = sim_df.sort_values("sim", ascending=False)
    return sim_df

def aggregate_sim_df(
        sim_df : "DataFrame", 
        top_k : int = 5
    ) -> "DataFrame":
    """
    Takes in a similarity dataframe and calculates the top_k neighbours based on the similarity, returning
    an aggregated dataframe.
    The top_k neighbours are the most similar k train smiles related to a test smile.
    Parameters:
        - sim_df : similarity dataframe
        - top_k : amount of k neighbours which should be considered for the aggregation on the sim score
    """
    sim_df = sim_df.sort_values(["sm1", "sim"], ascending=False)
    group_df = sim_df.groupby("sm1").head(top_k).sort_values("sm1").groupby("sm1").sim.agg(["min", "max", "mean", "std"]).sort_values("min")
    group_df = group_df.reset_index()
    group_df = group_df.rename({"sm1" : "smiles"}, axis=1)
    return group_df

def get_merged_df(
        group_df : "DataFrame", 
        pred_df : "DataFrame"
    ) -> "DataFrame":
    """
    Merging the grouped similarity DF with a prediction dataframe.
    Furthermore, the MAE is calculated and the mean similarity values are binned.
    Parameters:
        - group_df : Aggregated similarity df
        - pred_df : Predictions of a model containing smiles, activity and prediction
    """
    merged = group_df.merge(pred_df, on="smiles")
    merged["MAE"] = abs(merged.activity - merged.Preds)
    merged["Mean of top 5 similarity (Binned)"] = pd.cut(merged["mean"], bins=np.arange(0., 1.1, 0.1))
    merged = merged.sort_values("Mean of top 5 similarity (Binned)")
    return merged