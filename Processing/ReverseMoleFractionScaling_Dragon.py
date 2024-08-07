import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dataset_pt1 = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Log/MTrainLDm1.csv")
dataset_pt2 = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Log/MTrainLDm2.csv")
dataset_pt3 = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Log/MTrainLDm3.csv")
log_train_wo_dragon = pd.concat([dataset_pt1, dataset_pt2, dataset_pt3], axis=0).reset_index(drop=True)
log_test_wo_dragon = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quantile/MTestQDm.csv")
train_wo_dragon = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/WithoutDragon/master_train_unscaled_POC.csv")
test_wo_dragon = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/WithoutDragon/master_test_unscaled_POC.csv")

def reverse_log_scaling(log_scaled_series, original_series):
    min_val = float(original_series.min())
    shift = 0
    if min_val <= 0:
        shift = -min_val + 1e-7
    reversed_series = np.exp(log_scaled_series) - shift
    return reversed_series

# Reverse log scaling for all columns starting from column 6 and onwards
for column in log_train_wo_dragon.columns[6:7]:
    if column in train_wo_dragon.columns and column in test_wo_dragon.columns:
        log_train_wo_dragon[column] = reverse_log_scaling(log_train_wo_dragon[column], train_wo_dragon[column])
        log_test_wo_dragon[column] = reverse_log_scaling(log_test_wo_dragon[column], test_wo_dragon[column])

# Concatenate train and test sets
combined_data = pd.concat([log_train_wo_dragon, log_test_wo_dragon], axis=0)

# Apply MinMax scaling to columns starting from column 7 and onwards
min_max_scaler = MinMaxScaler()
columns_to_scale = combined_data.columns[7:]
combined_data[columns_to_scale] = min_max_scaler.fit_transform(combined_data[columns_to_scale])

# Splitting back to train and test datasets
log_train_wo_dragon_scaled = combined_data.iloc[:len(log_train_wo_dragon)]
log_test_wo_dragon_scaled = combined_data.iloc[len(log_train_wo_dragon):]

split_index = log_train_wo_dragon_scaled.shape[0]//5
train1 = log_train_wo_dragon_scaled.iloc[0:split_index, :]
train2 = log_train_wo_dragon_scaled.iloc[split_index:2*split_index, :]
train3 = log_train_wo_dragon_scaled.iloc[2*split_index:, :]

train1.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/LinearMF/Train_LinearMF_QMM1.csv", index=False)
train2.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/LinearMF/Train_LinearMF_QMM2.csv", index=False)
train3.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/LinearMF/Train_LinearMF_QMM3.csv", index=False)
log_test_wo_dragon_scaled.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/LinearMF/Test_LinearMF_QMM.csv", index=False)
