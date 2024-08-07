import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from temprange import minmax_calc, minmax_insert
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def log_scaling(dataset):
    dataset_scaled = dataset.copy()
    for col in dataset_scaled.columns:
        min_val = float(dataset_scaled[col].min())
        shift = 0
        if min_val <= 0:
            shift = -min_val + 1e-7  
        dataset_scaled[col] = np.log(dataset_scaled[col] + shift)

    dataset_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset_scaled.fillna(dataset_scaled.mean(), inplace=True)

    return dataset_scaled

def drop_single_value_columns(dataset):
    # This function checks each column and drops the ones where all values are the same
    for col in dataset.columns:
        if len(dataset[col].unique()) == 1:
            dataset.drop(col, inplace=True, axis=1)
    return dataset

def quantile_scaling(dataset):
    data = dataset.copy()
    cols = data.columns.tolist()
    transformer = QuantileTransformer(n_quantiles=1000, random_state=0, output_distribution='normal')
    scaled_data = pd.DataFrame(transformer.fit_transform(data))
    scaled_data.rename(columns={i:cols[i] for i in range(len(cols))}, inplace=True)

    return scaled_data

def minmax_scaling(dataset):
    data = dataset.copy()
    cols = data.columns.tolist()
    transformer = MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(transformer.fit_transform(data))
    scaled_data.rename(columns={i:cols[i] for i in range(len(cols))}, inplace=True)

    return scaled_data

def outlier_removal(dataset, n):
    with open("Datasets/TrainTestData/feature_rankings_processed.txt", "r") as file:
        feature_list = file.read()
    feature_list = feature_list.replace("[", "").replace("]", "").replace("'", "").split(", ")
    # cols = feature_list[0:n]
    cols = ["MoleFraction"]

    original_data = dataset.copy()

    for i in tqdm(range(len(cols))):
        try:
            col_vals = dataset[cols[i]].values
            mean = np.mean(col_vals)
            std = np.std(col_vals)

            for val in col_vals:
                if abs(val - mean) > 2.55*std: #2.55 for unscaled mole fracs, 2.0125 for scaled
                    dataset.drop(dataset[dataset[cols[i]] == val].index, inplace=True)
        except:
            pass
    outlier_set = pd.concat([original_data, dataset]).drop_duplicates(keep=False)
    return dataset, outlier_set

def printer(train, test):
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print("-"*70)

def resetter(df):
    df.drop_duplicates(inplace=True)

    with open("Datasets/IUPAC/FeatureEditing.json") as json_file: #Removing columns
        data = json.load(json_file)
    cols_to_remove = data['to remove']

    col_vals = df["MoleFraction"].values
    for val in col_vals:
        if val == 0 or val == 1: #Remove 0 mol frac values
            df.drop(df[df["MoleFraction"] == val].index, inplace=True)

    df.drop(cols_to_remove, axis=1, errors='ignore', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def main(name):
    SCALE_MF = False
    NO_DRAGON = True

    # train = resetter(pd.read_csv("Datasets/TrainTestData/MTrainUD_POC.csv"))
    # test = resetter(pd.read_csv("Datasets/TrainTestData/MTestUD_POC.csv"))
    train = resetter(pd.read_csv("Datasets/TrainTestData/Train2.csv"))
    test = resetter(pd.read_csv("Datasets/TrainTestData/Test2.csv"))

    df = pd.concat([train, test]) #Combining train and test sets

    if NO_DRAGON == True:
        df = df.iloc[:, 0:91]

    printer(train, test)
    train_shape = train.shape[0] #Saving train shape for later as split index

    minmax_dict = minmax_calc(df); print("Min and max temperatures calculated")

    if name == "Quantile": #Scaling of main block
        scaled_df = quantile_scaling(df.iloc[:, 6:])
        n = 70
    elif name == "Log": 
        scaled_df = log_scaling(df.iloc[:, 6:])
        n = 51
    elif name == "MinMax":
        scaled_df = minmax_scaling(df.iloc[:, 6:])
        n = 10
    elif name == "QuantileMinMax":
        scaled_df = quantile_scaling(df.iloc[:, 6:])
        scaled_df = minmax_scaling(scaled_df)
        n = 70
    elif name == "LogMinMax":
        scaled_df = log_scaling(df.iloc[:, 6:])
        scaled_df = minmax_scaling(scaled_df)
        n = 51
    else:
        scaled_df = df.iloc[:, 6:]
        n = 0
        name = "None"

    scaled_df = drop_single_value_columns(scaled_df)
    scaled_df = pd.concat([df.iloc[:, 0:6], scaled_df], axis=1)
    if name == "None":
        pass
    else:
        if SCALE_MF == True:
            raw_mole_fracs = df["MoleFraction"].values
            raw_mole_fracs = [float(i) for i in raw_mole_fracs]
            scaled_mole_fracs = np.log(raw_mole_fracs)
            scaled_df["MoleFraction"] = scaled_mole_fracs
            print("scaled mf")
        print(f"Dataframe {name} scaled")

    scaled_df = minmax_insert(minmax_dict, scaled_df); print("Min and max temperatures inserted")

    train = scaled_df.iloc[0:train_shape, :] #Splitting train and test sets
    test = scaled_df.iloc[train_shape:, :]
    printer(train, test)
    train.dropna(inplace=True); test.dropna(inplace=True) #Dropping NaNs
    train, outlier_set = outlier_removal(train, n) #Outlier removal from train set

    printer(train, test)
    print(f"Outlier shape: {outlier_set.shape}")

    if NO_DRAGON == True:
        train.to_csv(f"Datasets/TrainTestData/ImmiscibleSets/{name}Train.csv", index=False)
        test.to_csv(f"Datasets/TrainTestData/ImmiscibleSets/{name}Test.csv", index=False)
        outlier_set.to_csv(f"Datasets/TrainTestData/ImmiscibleSets/{name}Outliers.csv", index=False)
    else:
        split_index = train.shape[0]//3 #Splitting train set into 3 parts so GitHub doesn't throw a fit, and saving
        train1 = train.iloc[0:split_index, :]
        train2 = train.iloc[split_index:2*split_index, :]
        train3 = train.iloc[2*split_index:, :]

        train1.to_csv(f"Datasets/TrainTestData/Dragon/{name}Train1.csv", index=False)
        train2.to_csv(f"Datasets/TrainTestData/Dragon/{name}Train2.csv", index=False)
        train3.to_csv(f"Datasets/TrainTestData/Dragon/{name}Train3.csv", index=False)

        test.to_csv(f"Datasets/TrainTestData/Dragon/{name}Test.csv", index=False)
        outlier_set.to_csv(f"Datasets/TrainTestData/Dragon/{name}Outliers.csv", index=False)

if __name__ == "__main__":
    datasets = ["Quantile", "Log", "MinMax", "QuantileMinMax", "LogMinMax", "fdiaofjo"]
    for dataset in datasets:
        main(dataset)