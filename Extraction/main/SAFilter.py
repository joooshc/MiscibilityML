import os
import pandas as pd

def get_master_dataset_path():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.join(current_directory, '..','..', '..')
    dataset_directory = os.path.join(root_directory, 'Datasets')
    
    return os.path.join(dataset_directory, 'master_log_outliers_inc.csv')

def get_sigma_dataset_path():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.join(current_directory, '..', '..')
    dataset_directory = os.path.join(root_directory, 'Datasets', 'Sigma-Aldrich')
    
    return os.path.join(dataset_directory, 'Sigma-Aldrich_data.csv')

def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    return df

if __name__ == "__main__":
    master_dataset = load_dataset(get_master_dataset_path())
    master_compounds = master_dataset.iloc[:, :2]
    master_smiles = master_dataset.iloc[:, 2:4]

    sigma_dataset = load_dataset(get_sigma_dataset_path()).iloc[:, :6]
    sigma_compounds = sigma_dataset.iloc[:,:2]
    sigma_smiles = sigma_dataset.iloc[:,4:6]

    merged_smiles_data = pd.merge(master_dataset, sigma_smiles, 
                        left_on=master_smiles.columns.tolist(), 
                        right_on=sigma_smiles.columns.tolist(), 
                        how='inner')

    # Group by the first two columns and count the number of rows in each group
    grouped = merged_smiles_data.groupby([merged_smiles_data.columns[0], merged_smiles_data.columns[1]]).size().reset_index(name='count')
    repeated_pairs = grouped[grouped['count'] > 0]

