import pandas as pd
import numpy as np
from Evaluation import *
import os

os.makedirs("Imgs", exist_ok=True)

def get_result_df(endpoint, model, dataset, file):

    preds = []
    
    for num in range(1, 31): # Note: adjust to n_evals
        folder_path = f"Data/Confidence/{endpoint}/{model}/{dataset}/model_{num}"
        pred = pd.read_csv(f"{folder_path}/{file}_preds.csv")
        preds.append(pred[["Preds"]])
        if dataset != "Multitask":
            smiles = pred["smiles"].tolist()
        else:
            smiles = pd.read_csv(f"Data/Processed/Multitask/Test/{endpoint}_chemprop_test.csv")[f"activity_{file}"].dropna()
    
    preds = pd.concat(preds, axis=1)
    mean_pred = preds.mean(axis=1)
    target = pred.activity
    df = pd.DataFrame({
        "Preds" : mean_pred,
        "smiles" : smiles,
        "activity" : target
    })
        
    return df

for source in ["Intern", "Extern"]: # Adjust if needed


    plt.rcParams['xtick.labelsize'] = 13
    for _, model in enumerate(["Chemprop"]):
        
        
        fig, axs = plt.subplots(6, 2, figsize=(25, 25), dpi=550, sharey="row", sharex=True)
        for i, endpoint in enumerate(["hPPB", "rPPB", "Sol", "RLM", "HLM", "MDR1_ER"]):


            test_df = pd.read_csv(f"Data/Processed/{source}/Test/{endpoint}_chemprop_test.csv").iloc[:100, :]
            train_df = pd.read_csv(f"Data/Processed/{source}/Train/{endpoint}_chemprop_train.csv").iloc[:100, :]
            val_df = pd.read_csv(f"Data/Processed/{source}/Val/{endpoint}_chemprop_val.csv").iloc[:100, :]
            train_df = pd.concat((train_df, val_df), ignore_index=True)

            sim_df = get_sim_df(test_df, train_df)
            group_df = aggregate_sim_df(sim_df)
            for j, dataset in enumerate(["Intern", "Multitask"]):
                if dataset == "Intern" and source == "Intern":
                    dataset = "Extern"
                if dataset != "Multitask":
                    preds = get_result_df(endpoint, model, dataset, source.lower())
                else:
                    preds = pd.read_csv(f"Data/Evaluation/Chemprop/MT/{endpoint}_preds.csv")[["smiles", f"activity_{source.lower()}", f"{source} Preds"]].rename({f"activity_{source.lower()}" : "activity", f"{source} Preds" : "Preds"}, axis=1).dropna(axis=0, ignore_index=True)
                
                merged = get_merged_df(group_df, preds)

                cat_codes = merged["Mean of top 5 similarity (Binned)"].cat.codes
                merged["Mean of top 5 similarity (right Bin)"] = merged["Mean of top 5 similarity (Binned)"].apply(lambda x : x.right)
                sns.barplot(merged, 
                            x="Mean of top 5 similarity (right Bin)",
                            y="MAE",
                            errorbar="sd",
                            ax=axs[i, j],
                            color="royalblue")

                axs[i, j].title.set_text(f"{dataset} ({endpoint})")
                axs[i, j].title.set_size(20)
                axs[i, j].set_xlabel("")
                axs[i, j].set_ylabel("")
                axs[i, j].tick_params(axis='x', labelsize=17.5)  
                axs[i, j].tick_params(axis='y', labelsize=17.5)

        fig.tight_layout()
        fig.savefig(f"Imgs/{model}_{source}_Applic.png", bbox_inches="tight", dpi=550)