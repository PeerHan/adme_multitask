import pandas as pd
import xgboost
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from enum import Enum, auto
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConfidence(Enum):
    """Model confidence levels."""

    A = auto()
    B = auto()
    C = auto()
    D = auto()


CONFIDENCE_THRESHOLDS: tuple[float, ...] = (1.5, 3.0, 10.0)
ConfidenceModel = tuple[StandardScaler, LogisticRegression]

img_path = "Imgs"
os.makedirs(img_path, exist_ok=True)

def get_bounds(labels: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (min, max) bounds for the given confidence threshold.

    Args:
        labels: Float array with shope [num_values] containing experimental values. (Note that confidence levels are
            based on the likelihood of being within a threshold of the *true* value, not the predicted value.)
        threshold: Float confidence threshold.
        transform: Transform indicating the shape of the label space.

    Returns:
        min_values: Float array with shape [num_values] containing lower confidence bounds.
        max_values: Float array with shape [num_values] containing upper confidence bounds.
    """

    delta = np.log10(threshold)
    min_values = labels - delta
    max_values = labels + delta

    return min_values, max_values


def build_confidence_models(
    cv_results: pd.DataFrame, random_state: int
) -> dict[float, ConfidenceModel | None]:
    """Builds models used to predict confidence."""
    features = np.stack([cv_results["location"], cv_results["scale"]]).T
    assert features.shape == (len(cv_results), 2)
    confidence_models = {}
    for threshold in CONFIDENCE_THRESHOLDS:
        locations = cv_results["location"].values
        labels = cv_results["label"].values
        min_values, max_values = get_bounds(labels, threshold)
        confidence_labels = np.logical_and(locations >= min_values, locations <= max_values)
        scaler = StandardScaler()
        # NOTE(skearnes): Use penalty=None since regularization requires feature scaling.
        model = LogisticRegression(penalty=None, random_state=random_state)
        try:
            model.fit(scaler.fit_transform(features), confidence_labels)
            confidence_models[threshold] = (scaler, model)
        except ValueError as error:
            logger.debug(error)
            confidence_models[threshold] = None
    return confidence_models


def predict_confidence(predictions: pd.DataFrame, confidence_models: dict[float, ConfidenceModel]) -> np.ndarray:
    """Predicts confidence level probabilities.

    Args:
        predictions: DataFrame containing predictions.
        confidence_models: Dict mapping confidence thresholds to models.

    Returns:
        Numpy array with shape [num_molecules, num_confidence_levels] containing the probability for each
        confidence model.
    """
    features = np.stack([predictions["location"], predictions["scale"]]).T
    assert features.shape == (len(predictions), 2)
    confidence = np.zeros((len(predictions), len(confidence_models)), dtype=float)
    for i, threshold in enumerate(CONFIDENCE_THRESHOLDS):
        if confidence_models[threshold] is None:
            confidence[:, i] = np.nan
            continue
        scaler, model = confidence_models[threshold]
        confidence[:, i] = model.predict_proba(scaler.transform(features))[:, 1]
    return confidence


def assign_confidence(
    predictions: pd.DataFrame, confidence_models: dict[float, ConfidenceModel | None]
) -> pd.DataFrame:
    """Assigns confidence levels to predictions.

    Args:
        predictions: DataFrame containing predictions.
        confidence_models: Dict mapping confidence thresholds to models.

    Returns:
        DataFrame containing confidence probabilities and assignments.
    """
    confidence_predictions = predict_confidence(predictions, confidence_models=confidence_models)
    rows = []
    for confidence_row in confidence_predictions:
        row = {f"p(within {threshold:g}x)": confidence_row[j] for j, threshold in enumerate(CONFIDENCE_THRESHOLDS)}
        row["confidence_level"] = assign_confidence_level(confidence_row)
        rows.append(row)
    return pd.DataFrame(rows)


def assign_confidence_level(row: np.ndarray, threshold: float = 0.8) -> ModelConfidence:
    """Assigns a confidence level to predictions.

    Args:
        row: Numpy array with shape [num_confidence_levels] containing the probability for each confidence level.
        threshold: Probability threshold for assigning confidence levels.

    Returns:
        ModelConfidence assignment.
    """
    expected_shape = (len(CONFIDENCE_THRESHOLDS),)
    if row.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}; got {row.shape}")
    if row[0] >= threshold:
        return "A"
    if row[1] >= threshold:
        return "B"
    if row[2] >= threshold:
        return "C"
    return "D"

def get_result_df(endpoint, model, dataset, file):

    preds = []
    for num in range(1, 31): # NOTE : Adjust to amount of n_evals 
        folder_path = f"Data/Confidence/{endpoint}/{model}/{dataset}/model_{num}"
        pred = pd.read_csv(f"{folder_path}/{file}_preds.csv")
        preds.append(pred[["Preds"]])

    preds = pd.concat(preds, axis=1)
    location, scale = preds.mean(axis=1), preds.std(axis=1)
    target = pred.activity
    df = pd.DataFrame({"location" : location,
                       "scale" : scale,
                       "label" : target})
    return df

def get_test_target(test_df):

    A = np.logical_and(test_df.location >= test_df.label - np.log10(1.5), test_df.location <= test_df.label + np.log10(1.5))
    B = np.logical_and(test_df.location >= test_df.label - np.log10(3), test_df.location <= test_df.label + np.log10(3))
    C = np.logical_and(test_df.location >= test_df.label - np.log10(10), test_df.location <= test_df.label + np.log10(10))

    targets = []
    for i in range(len(A)):
        if A[i]:
            targets.append("A")
        elif B[i]:
            targets.append("B")
        elif C[i]:
            targets.append("C")
        else:
            targets.append("D")
    return targets

def get_conf_res(endpoint, model, train, test):
    if train == "intern_MT" or train == "extern_MT":
        prefix = train.split("_")[0]
        res_df = get_result_df(endpoint, model, "MT", f"{prefix}_train")
        train = "MT"
    else:
        res_df = get_result_df(endpoint, model, train, "train")
    logregs = build_confidence_models(res_df, 42)
    test_df = get_result_df(endpoint, model, train, test)
    df = assign_confidence(test_df.iloc[:, :-1], logregs)
    ocla_encoder = {"A" : 0, "B" : 1, "C" : 2, "D" : 3}
    df["Targets"] = get_test_target(test_df)
    df["Pred Encoded"] = df.confidence_level.map(ocla_encoder)
    df["Target Encoded"] = df.Targets.map(ocla_encoder)
    status = []
    for res in df["Pred Encoded"] - df["Target Encoded"]:
        if res == 0:
            status.append("Correct Conf.")
        elif res > 0:
            status.append("Overconf.")
        elif res < 0:
            status.append("Underconf.")
    df["Confidence"] = status
    return df, test_df
    
def concat_confidences(endpoint, model):
    train_sets = ["Intern", "Extern", "InEx"]
    if model == "Chemprop":
        train_sets.append("MT")
    res = []
    for test in ["extern", "intern"]:
        for train in train_sets:
            if train != "MT":
                res_df, test_df = get_conf_res(endpoint, model, train, test)
            else:
                res_df, test_df = get_conf_res(endpoint, model, f"{test}_MT", test)
            res_df["Dataset"] = train
            res_df["Test set"] = test
            res_cols = list(res_df.columns)
            res_df = pd.concat((res_df, test_df), axis=1, ignore_index=True)
            res_df.columns = res_cols + list(test_df.columns)
            res.append(res_df)
    res_df = pd.concat(res, ignore_index=True)
    return res_df

def ocla_acc(df):
    res = {}
    for ds in df.Dataset.unique():
        sub_df = df[df.Dataset == ds]
        n = len(sub_df)
        m = len(sub_df[sub_df.Confidence.isin(["Correct Conf.", "Underconf."])])
        res[ds] = m / n
    ocla_res = pd.DataFrame(res.items(), columns=["Training set", "OCLA Accuracy"])
    return ocla_res


sns.reset_orig()
data_map = {"Intern" : "Internal",
            "Extern" : "Public",
            "MT" : "Multitask",
            "InEx" : "Pooled"}
order_map = {"Internal" : 0, "Public" : 1, "Pooled" : 2, "Multitask" : 3}
plt.rc('legend',fontsize=25)
plt.rcParams['ytick.labelsize']=15
palette = "viridis_r"
fig, axs = plt.subplots(3, 2, sharex="col", sharey=0, dpi=150, figsize=(20, 5))

custom_dict = {'Extern': 0, 'Intern': 1, 'InEx': 2, "MT" : 3} 

colors = sns.color_palette(palette)
for i, endpoint in enumerate(["hPPB", "HLM", "MDR1_ER"]):
    
    res_df = concat_confidences(endpoint, "Chemprop")
    res_df.Dataset = res_df.Dataset.map(data_map)
    res_df["Test set"] = res_df["Test set"].str.capitalize().map(data_map)
    res_df["Order"] = res_df.Dataset.map(order_map)
    res_df = res_df.sort_values(["Order", "Test set"])

    intern = res_df[(res_df["Test set"] == "Internal") & (res_df.Dataset.isin(["Internal", "Multitask"]))]
    extern = res_df[(res_df["Test set"] == "Public") & (res_df.Dataset.isin(["Public", "Multitask"]))]   

    g1 = sns.histplot(intern,
                        y="Dataset",
                        hue="Confidence",
                        hue_order=["Correct Conf.", "Underconf.", "Overconf."],
                        palette=[colors[0], colors[1], colors[-2]],
                        multiple="fill",
                        legend=i==0,
                        ax=axs[i, 0])
    g1 = sns.histplot(extern,
                        y="Dataset",
                        hue="Confidence",
                        hue_order=["Correct Conf.", "Underconf.", "Overconf."],
                        palette=[colors[0], colors[1], colors[-2]],
                        multiple="fill",
                        legend=i==0,
                        ax=axs[i, 1])
    if i == 0:
        #sns.move_legend(axs[0, 1], "upper left", bbox_to_anchor=(1, 1))
        axs[0, 0].title.set_text("Internal Test set")
        axs[0, 0].title.set_size(20)
        axs[0, 1].title.set_text("Public Test set")
        axs[0, 1].title.set_size(20)
        for j in range(2):
            sns.move_legend(
                axs[i, j], "lower center",
                bbox_to_anchor=(.5, 1.4), ncol=4, title="Confidence Status", frameon=True,
                fontsize=15, title_fontsize=15
            )
    axs[i, 0].set_ylabel(f"{endpoint}", size=12.5)
    axs[i, 1].set_ylabel("")
    axs[i, 0].set_xlabel("")
    axs[i, 1].set_xlabel("")
    
    axs[i, 0].axvline(0.2, color="red", ls="--")
    axs[i, 1].axvline(0.2, color="red", ls="--")

fig.suptitle("Ordinal Confidencelevel Assignment on MPNN Models", y=0.95, size=25)
fig.supxlabel("Percentage", size=20)
fig.tight_layout()
fig.savefig(f"{img_path}/Chemprop_OCLA_Plot_{palette}.png",
            bbox_inches="tight",
            dpi=250)

sns.reset_orig()
data_map = {"Intern" : "Internal",
            "Extern" : "Public",
            "MT" : "Multitask",
            "InEx" : "Pooled"}
order_map = {"Internal" : 0, "Public" : 1, "Pooled" : 2, "Multitask" : 3}
plt.rc('legend',fontsize=25)
plt.rcParams['ytick.labelsize']=15
for model in ["XGB", "Chemprop", "MLP"]:
    for palette in ["viridis_r"]:
        fig, axs = plt.subplots(6, 2, sharex=True, sharey=0, dpi=150, figsize=(20, 20))

        #sns.set(font_scale=2)  # crazy big
        custom_dict = {'Extern': 0, 'Intern': 1, 'InEx': 2, "MT" : 3} 
        colors = sns.color_palette(palette)


        for i, endpoint in enumerate(["hPPB", "rPPB", "Sol", "RLM", "HLM", "MDR1_ER"]):

            res_df = concat_confidences(endpoint, model)
            res_df.Dataset = res_df.Dataset.map(data_map)
            res_df["Test set"] = res_df["Test set"].str.capitalize().map(data_map)
            res_df["Order"] = res_df.Dataset.map(order_map)
            res_df = res_df.sort_values(["Order", "Test set"])
            endpoint = "MDR1-MDCK ER" if endpoint == "MDR1_ER" else endpoint

            intern = res_df[(res_df["Test set"] == "Internal")]
            extern = res_df[(res_df["Test set"] == "Public")] 

            g1 = sns.histplot(intern,
                              y="Dataset",
                              hue="Confidence",
                              hue_order=["Correct Conf.", "Underconf.", "Overconf."],
                              palette=[colors[0], colors[1], colors[-2]],
                              multiple="fill",
                              legend=i==0,
                              ax=axs[i, 0])
            g1 = sns.histplot(extern,
                              y="Dataset",
                              hue="Confidence",
                              hue_order=["Correct Conf.", "Underconf.", "Overconf."],
                              palette=[colors[0], colors[1], colors[-2]],
                              multiple="fill",
                              legend=i==0,
                              ax=axs[i, 1])
            if i == 0:
                #sns.move_legend(axs[0, 1], "upper left", bbox_to_anchor=(1, 1))
                axs[0, 0].title.set_text("Internal Test set")
                axs[0, 0].title.set_size(20)
                axs[0, 1].title.set_text("Public Test set")
                axs[0, 1].title.set_size(20)
                for j in range(2):
                    sns.move_legend(
                        axs[i, j], "lower center",
                        bbox_to_anchor=(.5, 1.15), ncol=4, title="Confidence Status", frameon=True,
                        fontsize=15, title_fontsize=15
                    )
            axs[i, 0].set_ylabel(f"{endpoint}", size=19)
            axs[i, 1].set_ylabel("")
            axs[i, 0].set_xlabel("")
            axs[i, 1].set_xlabel("")

            axs[i, 0].axvline(0.2, color="red", ls="--")
            axs[i, 1].axvline(0.2, color="red", ls="--")

        fig.suptitle(f"Ordinal Confidencelevel Assignment of {model if model != 'Chemprop' else 'MPNN'} Models", y=1, size=25)
        fig.supxlabel("Percentage", size=20)
        fig.tight_layout()
        fig.savefig(f"{img_path}/OCLA_Plot_{palette}_{model}.png",
                        bbox_inches="tight",
                        dpi=250)