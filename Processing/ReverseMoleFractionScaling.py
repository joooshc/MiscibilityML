import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_data(paths):
    return [pd.read_csv(path) for path in paths]

def apply_reverse_log_scaling(log_df, orig_df):
    for column in log_df.columns[6:]:
        if column != 'Temperature' and column in orig_df.columns:
            log_df[column] = reverse_log_scaling(log_df[column], orig_df[column])
    return log_df

def reverse_log_scaling(log_scaled_series, original_series):
    min_val = float(original_series.min())
    shift = 0
    if min_val <= 0:
        shift = -min_val + 1e-7
    return np.exp(log_scaled_series) - shift

def scale_data(df):
    min_max_scaler = MinMaxScaler()
    print(df)
    quit()
    columns_to_scale = df.columns[7:]
    df[columns_to_scale] = min_max_scaler.fit_transform(df[columns_to_scale])
    return df

def remove_columns(df, cols_to_remove):
    return df.drop(columns=cols_to_remove)

def main():
    paths = [
        "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragon/MTrainLDm.csv",
        "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragon/MTestLDm.csv",
        "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/MTrainUD_POC.csv",
        "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/MTestUD_POC.csv"
    ]
    log_train, log_test, train, test = read_data(paths)
    to_remove = ["MW", "AMW", "Sv", "NumHAcceptors", "NumHeteroatoms", "NumRotatableBonds", 
             "RingCount", "RotatableBondCount", "FeatureCount3D", "EffectiveRotorCount3D",
             "BCUT2D_MWLOW", "PEOE_VSA1", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8",
             "SMR_VSA1", "SMR_VSA10", "SMR_VSA5", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA5",
             "EState_VSA4", "EState_VSA5", "EState_VSA8", "EState_VSA9", "VSA_EState3", 
             "VSA_EState5", "VSA_EState7", "VSA_EState8", "FractionCSP3", "NOCount"]
    
    log_train = apply_reverse_log_scaling(log_train, train)
    log_test = apply_reverse_log_scaling(log_test, test)
    
    combined_data = pd.concat([log_train, log_test], axis=0)
    
    scaled_data = scale_data(combined_data)
    
    log_train_scaled = scaled_data.iloc[:len(log_train)]
    log_test_scaled = scaled_data.iloc[len(log_train):]
    
    log_train_scaled = remove_columns(log_train_scaled, to_remove)
    log_test_scaled = remove_columns(log_test_scaled, to_remove)
    
    log_train_scaled.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/ReducedFeatures/WithoutDragon/Train_LinearMF_RMM.csv", index=False)
    log_test_scaled.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/ReducedFeatures/WithoutDragon/Test_LinearMF_RMM.csv", index=False)

if __name__ == "__main__":
    main()