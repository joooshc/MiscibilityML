import os, json, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

def extract():
    """
    Extracts training and test data from a given dataset. 

    The function retrieves the path of the dataset based on the current 
    script's location, imports the dataset, groups by SMILES codes, and 
    creates the training and test datasets. The datasets are then saved 
    as CSV files.

    Parameters:
    None

    Returns:
    tuple: Two pd.DataFrames
        - training_data (pd.DataFrame): The constructed training data.
        - test_data (pd.DataFrame): The constructed test data.
    """

    # Retrieving paths for the datasets used in this script
    py_file_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.dirname(os.path.dirname(py_file_path))
    datasets_path = os.path.join(root_dir_path, "Datasets")

    # Importing the datasets
    os.chdir(datasets_path)
    master_unscaled = pd.read_csv("master_unscaled_dragon.csv")
    master_compounds = master_unscaled.iloc[:, :2] # not using atm
    master_smiles = master_unscaled.iloc[:, 2:4]

    # Group by the SMILES codes and count the number of occurances for each unique compound
    grouped = master_unscaled.groupby([master_smiles.columns[0], master_smiles.columns[1]]).size().reset_index(name='count')
    repeated_pairs = grouped[grouped['count'] > 0]

    # Calculate and remove 20% of the total repeated pairs
    num_to_remove = int(0.2 * len(repeated_pairs))
    test_set = repeated_pairs.sample(num_to_remove)
    training_set = repeated_pairs.drop(test_set.index)

    # Select rows from master_unscaled with equivalent SMILES1 and SMILES2 in training_set
    training_data = pd.merge(master_unscaled, training_set[[master_smiles.columns[0], master_smiles.columns[1]]], 
                            on=[master_smiles.columns[0], master_smiles.columns[1]], 
                            how='inner')
    test_data = pd.merge(master_unscaled, test_set[[master_smiles.columns[0], master_smiles.columns[1]]], 
                        on=[master_smiles.columns[0], master_smiles.columns[1]], 
                        how='inner')

    print(f"Trainig set and Test set shapes: {training_set.shape}, {test_set.shape}")
    print(f"Training data shape: {training_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    return training_data, test_data, datasets_path

def remove_features():
    """
    Remove columns where all elements are only a specific integer (e.g., 0 or 1).
    Remove outliers based on IQR.
    
    Parameters:
    None

    Returns:
    - pd.DataFrame, pd.DataFrame: Updated training and test data.
    """
    training_data, test_data, datasets_path = extract()
    iupac_path = os.path.join(datasets_path, "IUPAC")
    
    with open(os.path.join(iupac_path, 'FeatureEditing.json'), 'r') as file:
        feature_editing = json.load(file)

    # Combine training and test data
    combined_data = pd.concat([training_data, test_data], axis=0).reset_index(drop=True)

    # Remove columns specified in the json
    columns_to_remove = feature_editing.get('to remove', [])
    combined_data = combined_data.drop(columns=[col for col in columns_to_remove if col in combined_data.columns])

    # Define a function to check if all values in a series are the same
    def all_same(series):
        return series.nunique() == 1

    # Function to check if more than 50% of a series' values are the same
    def is_sparse(series):
        most_common_value_count = series.value_counts().iloc[0]
        return most_common_value_count > 0.5 * len(series)

    # Drop columns where all values are the same
    cols_to_drop = combined_data.columns[combined_data.apply(all_same, axis=0)]
    combined_data = combined_data.drop(columns=cols_to_drop)

    # Get the starting index for checking sparse columns
    starting_index = combined_data.columns.get_loc("SMILES2") + 1

    # Drop sparse columns (only after the 'SMILES2' column)
    sparse_cols = combined_data.columns[starting_index:][combined_data.iloc[:, starting_index:].apply(is_sparse, axis=0)]
    combined_data = combined_data.drop(columns=sparse_cols)
    if 'SMILES1.1' in combined_data.columns:
        combined_data = combined_data.drop(columns=['SMILES1.1'])

    # Drop NaN values
    combined_data.dropna(inplace=True)

    # Check and print the number of NaNs after dropping them
    nans = combined_data.isna().sum().sum()
    print(f"Number of NaNs in data after dropping: {nans}")
    
    # Split the data back into training and test sets based on their lengths
    training_data = combined_data.iloc[:len(training_data), :]
    test_data = combined_data.iloc[len(training_data):, :]

    training_data.to_csv("MTrainUD_POC.csv", index=False)  # master_train_unscaled_dragon
    test_data.to_csv("MTestUD_POC.csv", index=False)      # master_test_unscaled_dragon

    return training_data, test_data

def log_scale():
    """
    Apply log scaling to the input dataframes after combining them.
    
    Parameters:
    None

    Returns:
    - pd.DataFrame, pd.DataFrame: Log-scaled training and test data.
    """
    training_data, test_data = remove_features()

    # Concatenate the datasets
    combined_data = pd.concat([training_data, test_data])

    def log_scaling(dataset):
        dataset_scaled = dataset.copy()

        for col in dataset_scaled.columns:
            min_val = dataset_scaled[col].min()
            shift = 0
            if min_val <= 0:
                shift = -min_val + 1e-7
            dataset_scaled[col] = np.log(dataset_scaled[col] + shift)

        dataset_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset_scaled.fillna(dataset_scaled.mean(), inplace=True)

        return dataset_scaled

    # Apply log scaling to combined dataset
    combined_data_scaled_values = log_scaling(combined_data.iloc[:, 4:])

    # Convert transformed values back to DataFrame
    combined_data_scaled = pd.DataFrame(data=combined_data_scaled_values, 
                                        columns=combined_data.columns[4:],
                                        index=combined_data.index)

    # Split back into training and test datasets
    training_data_scaled = pd.concat([training_data.iloc[:, :4], combined_data_scaled.loc[training_data.index]], axis=1)
    test_data_scaled = pd.concat([test_data.iloc[:, :4], combined_data_scaled.loc[test_data.index]], axis=1)

    # Check and print the number of NaNs
    train_nans = training_data_scaled.isna().sum().sum()
    test_nans = test_data_scaled.isna().sum().sum()
    print(f"Number of NaNs in training data after log scaling: {train_nans}")
    print(f"Number of NaNs in test data after log scaling: {test_nans}")

    training_data_scaled.to_csv("master_train_log_POC.csv", index=False)
    test_data_scaled.to_csv("master_test_log_POC.csv", index=False)

    return training_data_scaled, test_data_scaled

def quantile_scale():
    """
    Apply quantile scaling to the input dataframes.
    
    Parameters:
    None

    Returns:
    - pd.DataFrame, pd.DataFrame: Quantile-scaled training and test data.
    """
    training_data, test_data = remove_features()

    # Concatenate the datasets
    combined_data = pd.concat([training_data, test_data])

    # Define and fit the quantile transformer
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
    combined_data_scaled_values = quantile_transformer.fit_transform(combined_data.iloc[:, 4:])

    # Convert transformed values back to DataFrame
    combined_data_scaled = pd.DataFrame(data=combined_data_scaled_values, 
                                        columns=combined_data.columns[4:],
                                        index=combined_data.index)

    # Split back into training and test datasets
    training_data_scaled = pd.concat([training_data.iloc[:, :4], combined_data_scaled.loc[training_data.index]], axis=1)
    test_data_scaled = pd.concat([test_data.iloc[:, :4], combined_data_scaled.loc[test_data.index]], axis=1)

    # Check and print the number of NaNs
    train_nans = training_data_scaled.isna().sum().sum()
    test_nans = test_data_scaled.isna().sum().sum()
    print(f"Number of NaNs in training data after quantile scaling: {train_nans}")
    print(f"Number of NaNs in test data after quantile scaling: {test_nans}")

    training_data_scaled.to_csv("master_train_quantile_POC.csv", index=False)
    test_data_scaled.to_csv("master_test_quantile_POC.csv", index=False)

    return training_data_scaled, test_data_scaled

_,_ = remove_features()