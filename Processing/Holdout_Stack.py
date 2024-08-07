import pandas as pd
import numpy as np
import time
from tqdm import tqdm

def train_test_gen():
    # Defining csv file names
    input_folders = ["NoDragon", "NoDragonY", "NoDragonYx"]
    output_folders = ["NoDragon-stacking", "NoDragonY-stacking", "NoDragonYx-stacking"]
    train_file_names = ["LogMinMaxTrain", "LogTrain", "MinMaxTrain", "QuantileMinMaxTrain", "QuantileTrain"]
    test_file_names = ["LogMinMaxTest", "LogTest", "MinMaxTest", "QuantileMinMaxTest", "QuantileTest"]

    # Defining in and out datapaths
    file_path_in = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData"
    file_path_out = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Extra"

    # Looping over each train and test dataset
    for folder_in, folder_out in zip(input_folders, output_folders):
        for train_file_name, test_file_name in tqdm(zip(train_file_names, test_file_names)):
            train = pd.read_csv(f"{file_path_in}/{folder_in}/{train_file_name}.csv")
            test = pd.read_csv(f"{file_path_in}/{folder_in}/{test_file_name}.csv")
            
            # Concatenate the train and test set and combining compound pairs
            combined_pairs = pd.DataFrame()
            combined_pairs['CompoundPair'] = train['Compound1'] + "-" + train['Compound2']

            # Dropping duplicates based on this new 'CompoundPair' column to get unique pairs
            unique_pairs = combined_pairs.drop_duplicates()

            # Filtering out unique pairs from the combined dataset
            unique_pairs_80 = unique_pairs.sample(frac=0.8, random_state=42)
            unique_pairs_20 = unique_pairs.loc[~unique_pairs.index.isin(unique_pairs_80.index)]
            train['CompoundPair'] = combined_pairs

            # Separate the train and test sets
            new_train_dataset = train[train['CompoundPair'].isin(unique_pairs_80['CompoundPair'])].iloc[:, :-4]
            '''new_train_dataset = oversample(new_train_dataset)''' # Uncomment to oversample
            new_test_dataset_1 = train[train['CompoundPair'].isin(unique_pairs_20['CompoundPair'])].iloc[:, :-4]
            new_test_dataset_2 = test.iloc[:, :-3]

            # Saving new datasets
            new_train_dataset.to_csv(f"{file_path_out}/{folder_out}/{train_file_name}-stack.csv",index=False)
            new_test_dataset_1.to_csv(f"{file_path_out}/{folder_out}/{test_file_name}1-stack.csv",index=False)
            new_test_dataset_2.to_csv(f"{file_path_out}/{folder_out}/{test_file_name}2-stack.csv",index=False)

# Oversample underrepresented mole fractions to about 40% of the dataset
def oversample(new_train_dataset):
    to_oversample = new_train_dataset[new_train_dataset['MoleFraction'] > 0.1]
    if to_oversample.empty:
        return new_train_dataset
    oversampled = to_oversample.sample(n=int(new_train_dataset.shape[0]*0.8 - to_oversample.shape[0]), replace=True)
    return pd.concat([new_train_dataset, oversampled], axis=0).reset_index(drop=True)

def get_inverse_temps():
    def apply_reverse_log_scaling(log_df, orig_df):
        for column in log_df.columns[6:-2]:
            log_df[column] = reverse_log_scaling(log_df[column], orig_df[column])
        return log_df

    def reverse_log_scaling(log_scaled_series, original_series):
        min_val = float(original_series.min())
        shift = 0
        if min_val <= 0:
            shift = -min_val + 1e-7
        return np.exp(log_scaled_series) - shift

    paths = [
        "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Extra/NoDragonYx-stacking/LogTrain-stack.csv",
        "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Extra/NoDragonYx-stacking/LogTest1-stack.csv",
        "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Extra/NoDragonYx-stacking/LogTest2-stack.csv",
        "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/MTrainUD_POC.csv",
        "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/MTestUD_POC.csv"
    ]
    log_train, log_test1, log_test2, train, test = [pd.read_csv(path) for path in paths]
    original_dataset = pd.concat([train, test], axis=0)
    log_train = apply_reverse_log_scaling(log_train, original_dataset)
    log_test1 = apply_reverse_log_scaling(log_test1, original_dataset)
    log_test2 = apply_reverse_log_scaling(log_test2, original_dataset)

    return log_train['Temperature'], log_test1['Temperature'], log_test2['Temperature']

def create_results_set(test1_temps, test2_temps):
    # Defining in and out datapaths
    file_path_in1 = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Extra/NoDragon-stacking"
    file_path_in2 = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/RF/RandomForest-Stacking/RF_ResultsYx_Test1"
    file_path_in3 = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/RF/RandomForest-Stacking/RF_ResultsYx_Test2"
    file_path_out = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/Stacking"

    # Defining file names:
    test_sets = ["LogTest1-stack", "LogTest2-stack"]
    y_preds = ["LogMinMaxTest", "LogTest", "MinMaxTest", "QuantileMinMaxTest", "QuantileTest"]

    # Looping over entries in file path 1
    test_set_lst = []
    for test_set in test_sets:
        test_sets = pd.read_csv(f"{file_path_in1}/{test_set}.csv").iloc[:, :7]
        test_set_lst.append(test_sets)
    
    test_set1 = pd.concat([test_set_lst[0], test1_temps], axis=1)
    test_set2 = pd.concat([test_set_lst[1], test2_temps], axis=1)
    
    # Looping over entries in file path 1
    for y_pred in y_preds:
        y1 = pd.read_csv(f"{file_path_in2}/{y_pred}.txt", sep=',', header=None).T[0]
        y2 = pd.read_csv(f"{file_path_in3}/{y_pred}.txt", sep=',', header=None).T[0]
        test_set1[f"DNN-{y_pred}"] = y1
        test_set2[f"DNN-{y_pred}"] = y2

    test_set1.to_csv(f"{file_path_out}/TrainYx_rf.csv", index=False)
    test_set2.to_csv(f"{file_path_out}/TestYx_rf.csv", index=False)

if __name__ == "__main__":
    # To generate stacking train and test sets:
    '''    
    train_test_gen()
    '''

    # To create the results_dataset for test1:
    train_temps, test1_temps, test2_temps = get_inverse_temps()
    create_results_set(test1_temps, test2_temps)
