#!/usr/bin/env python3
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
import os

def standardize(
        smiles : str,
        return_smiles : bool = True
    ) -> "Mol | SMILES":
    """
    Function to standardize SMILES based on their Mol representation.
    Taken and adapted from https://www.blopig.com/blog/2022/05/molecular-standardization/
    Standardization involves:
        - Cleaning : RemoveHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        - Fragmenting : If many fragments, get the "parent" (the actual mol we are interested in) 
        - Uncharge : Try to neutralize molecule
        - Tautomers : Canonicalize tautomers
    
    Parameters:
        - smiles : SMILES to standardize
        - return_smiles : Returns standardized SMILES instead of Mol
    """
    mol = Chem.MolFromSmiles(smiles)
    clean_mol = rdMolStandardize.Cleanup(mol) 
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()
    mol_final = te.Canonicalize(uncharged_parent_clean_mol)
    if return_smiles:
        return Chem.MolToSmiles(mol_final, kekuleSmiles=True)
    else:
        return mol_final

def featurize(
        df : "DataFrame", 
        target : str = "activity", 
        id_col : str = "ID",
        standardize_mol : bool = False,
    ) -> (list, list, list, list):
    """
    Function to featurize ROMol objects with various 2D rdkit descriptors and morgan fingerprint 
    with a length of 1024. 

    Taken and adapted from:
    https://github.com/molecularinformatics/Computational-ADME
    
    Parameters:
        - df : Dataframe with column 'smiles' and 'activity' (target)
        - target : Name for the target
        - id_col : Col for ID - if not provided a numerical index will be created
        - standardize_mol : If true, each mol will be standardized
    """
    df = df.copy()
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol="smiles")
    if not (id_col in df.columns):
        df[id_col] = [num for num in range(len(df))]
    
    act = {}
    fcfp4_bit = {}
    rdMD = {}
    name_list = []

    for _, row in df.iterrows():
        mol = row.ROMol
        if standardize_mol:
            mol = standardize(mol)
        Chem.rdDepictor.Compute2DCoords(mol)
        mol_name = row[id_col]
        name_list.append(mol_name)
        activity = row[target]
        act[mol_name] = float(activity)

        MDlist = []
        try:
            MDlist.append(rdMolDescriptors.CalcTPSA(mol))
            MDlist.append(rdMolDescriptors.CalcFractionCSP3(mol))
            MDlist.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
            MDlist.append(rdMolDescriptors.CalcNumAmideBonds(mol))
            MDlist.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumAromaticRings(mol))
            MDlist.append(rdMolDescriptors.CalcNumLipinskiHBA(mol))
            MDlist.append(rdMolDescriptors.CalcNumLipinskiHBD(mol))            
            MDlist.append(rdMolDescriptors.CalcNumHeteroatoms(mol))
            MDlist.append(rdMolDescriptors.CalcNumRings(mol))
            MDlist.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
            MDlist.append(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
            MDlist.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
            MDlist.append(rdMolDescriptors.CalcHallKierAlpha(mol))
            MDlist.append(rdMolDescriptors.CalcKappa1(mol))
            MDlist.append(rdMolDescriptors.CalcKappa2(mol))
            MDlist.append(rdMolDescriptors.CalcKappa3(mol))
            MDlist.append(rdMolDescriptors.CalcChi0n(mol))
            MDlist.append(rdMolDescriptors.CalcChi0v(mol))
            MDlist.append(rdMolDescriptors.CalcChi1n(mol))
            MDlist.append(rdMolDescriptors.CalcChi1v(mol))
            MDlist.append(rdMolDescriptors.CalcChi2n(mol))
            MDlist.append(rdMolDescriptors.CalcChi2v(mol))
            MDlist.append(rdMolDescriptors.CalcChi3n(mol))
            MDlist.append(rdMolDescriptors.CalcChi3v(mol))
            MDlist.append(rdMolDescriptors.CalcChi4n(mol))
            MDlist.append(rdMolDescriptors.CalcChi4v(mol))
            MDlist.append(rdMolDescriptors.CalcAsphericity(mol))
            MDlist.append(rdMolDescriptors.CalcEccentricity(mol))   
            MDlist.append(rdMolDescriptors.CalcInertialShapeFactor(mol))
            MDlist.append(rdMolDescriptors.CalcExactMolWt(mol))  
            MDlist.append(rdMolDescriptors.CalcPBF(mol))  
            MDlist.append(rdMolDescriptors.CalcPMI1(mol))
            MDlist.append(rdMolDescriptors.CalcPMI2(mol))
            MDlist.append(rdMolDescriptors.CalcPMI3(mol))
            MDlist.append(rdMolDescriptors.CalcRadiusOfGyration(mol))
            MDlist.append(rdMolDescriptors.CalcSpherocityIndex(mol))
            MDlist.append(rdMolDescriptors.CalcLabuteASA(mol))
            MDlist.append(rdMolDescriptors.CalcNPR1(mol))
            MDlist.append(rdMolDescriptors.CalcNPR2(mol))
            for d in rdMolDescriptors.PEOE_VSA_(mol): 
                MDlist.append(d)
            for d in rdMolDescriptors.SMR_VSA_(mol): 
                MDlist.append(d)
            for d in rdMolDescriptors.SlogP_VSA_(mol): 
                MDlist.append(d)
            for d in rdMolDescriptors.MQNs_(mol): 
                MDlist.append(d)
            for d in rdMolDescriptors.CalcCrippenDescriptors(mol):
                MDlist.append(d)
            for d in rdMolDescriptors.CalcAUTOCORR2D(mol):  
                MDlist.append(d)
        except:
            print ("The RDdescritpor calculation failed!")

        rdMD[mol_name] = MDlist

        try:
            fcfp4_bit_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=1024)
            fcfp4_bit[mol_name] = fcfp4_bit_fp.ToBitString()
        except:
            fcfp4_bit[mol_name] = ""
            print ("The FCFP4 calculation failed!")                
    
    return act, fcfp4_bit, rdMD, name_list

def create_featurized_df(
        df : "DataFrame", 
        target : str = "activity", 
        save_path : str = None,
        id_col : str = "ID",
        drop_id : bool = True
    ) -> "DataFrame":
    """
    Function to featurize all smiles of a dataset and return a featurized dataframe.
    
    Parameters:
        - df : Dataframe with at least 2 columns : smiles, activity
        - target : Column for the target
        - save_path : If specified, the result dataframe will be saved there
        - id_col : Column for the ID 
    """
    
    act, fcfp4, rdMD, name_list = featurize(df, id_col = id_col)
    header = [target]
    data = {}

    for i, key in enumerate(name_list):
        start = (i == 0)
        name = key
        tmp = []
        activity = act[name]
        fcfp4D = list(fcfp4[name])
        tmp.append(activity)
    
        k = 1
        for fp_val in fcfp4D:
            tmp.append(fp_val)
            if start:
                varname = f"fcfp4_{k}"
                header.append(varname)
                k += 1
        
        rdMD_des = rdMD[name]
        k = 1
        for descriptor in rdMD_des:
            tmp.append(descriptor)
            if start:
                varname = f"rdMD_{k}d"
                header.append(varname)
                k += 1

        data[key] = tmp

    
    df = pd.DataFrame.from_dict(data, orient="index", columns=header)
    df = df.astype(float)
    df = df.reset_index(names="ID")
    if drop_id:
        df = df.drop(["ID"], axis=1)
    if save_path:
        df.to_csv(save_path, index=False)
    return df