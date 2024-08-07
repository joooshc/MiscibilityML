import warnings
import pandas as pd
from Preprocess import Preprocessing
from CASManager import CASManager
import numpy as np
import os, time

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # FLAG DECLARATIONS FOR DEBUGGING
    DAN = True
    JOSH = False

    if DAN:
        # INTERNAL FLAG DECLARATIONS
        PLOT_HIST = False  
        PLOT_SCALED_HIST = False 

        # Extract Features and build the raw dataset
        cas_manager = CASManager()
        cas_dict, df = cas_manager.smiles_gen()
        cas_pubchem_props_dict = cas_manager.pubchem_props_gen(cas_dict)
        cas_rdkit_descriptors_dict = cas_manager.rdkit_descriptors_gen(cas_dict)
        df1, df2, df_diff = cas_manager.merge_dicts_to_df(df, cas_pubchem_props_dict, cas_rdkit_descriptors_dict)

        # Preprocess by removing columns with all 0's and 1's and log scale the data
        file_path = "diff_dataset_log_scaled.csv"
        if os.path.exists(file_path):
            dataset_log = pd.read_csv(file_path)
        else:
            dataset_log = pd.read_csv("Datasets/DDBprocesseddata/y1/diff_dataset_raw.csv")
            Preprocessing.process_dataset(dataset_log, 20, PLOT_HIST, PLOT_SCALED_HIST, "FeatureEditing.json")

    if JOSH: # For Josh's code
        pass