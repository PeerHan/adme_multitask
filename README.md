# ADME Multitask
Accurate predictions of compound properties are crucial for enhancing drug discovery by expediting processes and increasing success rates. This study focuses on predicting key pharmacokinetic endpoints related to Absorption, Distribution, Metabolism, and Excretion (ADME), leveraging extensive internal and newly available public hiqh-quality ADME data. We assess data-integration strategies for ADME prediction across six endpoints using single‑source (internal or public), pooled single‑task, and multi‑task learning models. 
Models trained on combined data — especially multi-task models — generally outperform single‑source baselines, with consistent gains on public tests and frequent gains on internal tests when public data complement and are proportionally balanced with in‑house data size. Applicability domain analyses show that multi-task learning reduces error for compounds with higher similarity to the training space, indicating better generalization across combined spaces. Analysis of prediction uncertainties mirrors these observations. Our study underscores that curated integration of high‑quality public datasets with proprietary data can deliver more accurate and better‑calibrated in silico ADME models to support computational compound design in drug discovery.

## Citation

## Get Started

### Create Env
- Create a conda environment: `conda create --name my_env --file requirements.txt`

### Data

#### Create Data splits
- Data/Unprocessed contains the training (Extern_Unsplit) and test (Extern_Test) files from [Fang et al.](https://pubs.acs.org/doi/full/10.1021/acs.jcim.3c00160) and the corresponding [git repository](https://github.com/molecularinformatics/Computational-ADME/tree/main)
- Training and validation splits are provided by executing `python3 scripts/split_extern_data.py` resulting in the respective folders Extern_Train and Extern_Val 
- Analogous Intern data should be provided in Data/Unprocessed as 'Intern_Train', 'Intern_Val', and 'Intern_Test' with the SMILES column 'smiles' and target column 'activity'

#### Standardize and Combine Data
- Data is standardized, deduplicated, and pooled across data sources with executing `python3 scripts/data_processing.py`
- Processed data is in the new created folders 'Processed/Intern', 'Processed/Extern', and 'Processed/InEx'
- The target column units might be changed depending on the internal data
- A folder 'Processed/Merged' contains information about duplicates across the data sources
- The multitask datasets are created by executing `python3 scripts/create_mt_datasets.py`
- The Processed folder now contains the multi-task datasets suitable for chemprop models in the Subfolder 'Processed/Multitask'

### Model Training

#### Hyperparameter Tuning
- Start the hyperparameter tuning with `python3 scripts/hp_search_ml.py --endpoint e --trials t --dataset d` for n trials and an endpoint e. A dataset d can be specified, per default all datasets are used. 
- An analogous script is provided for the multi-task experiment and can be executed with `python3 scripts/hp_search_mt.py --endpoint e --trials t` for t trials and an endpoint e 
- The folder 'Data/HP_Results' contains logs in the subfolder 'Logs' and the results of the HP tuning besides the study object in the subfolder 'ML'

#### Re-train and Evaluate
- Execute `python3 scripts/train_and_evaluate_model.py --endpoint e --n_evals n --model m` for model m on endpoint e with n diferent random seeds 
- The folder 'Data/Evaluation' contains the test predictions and the test metrics for each respective model
- The folder 'Data/Confidence' contains the training data (training prediction) for OCLA

#### Applicability Domain
- Execute `python3 scripts/applicability_domain.py` to assess the applicability domain of certain models depending on the training set
- The results will be saved in an Img folder 

#### OCLA 
- Execute `python3 scripts/OCLA.py` 
- OCLA Plots will be saved in a folder called Imgs for all endpoints and all algorithms, the amount of n_evals must be set accordingly
