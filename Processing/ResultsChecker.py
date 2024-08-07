import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def reverse_log_scaling(log_scaled_series, original_series):
    min_val = float(original_series.min())
    shift = 0
    if min_val <= 0:
        shift = -min_val + 1e-7
    reversed_series = np.exp(log_scaled_series) - shift
    return reversed_series

def metrics(y, y_preds):
    for i in range(y_preds.shape[1]):
        pred_column = y_preds.iloc[:, i]
        r2 = r2_score(y, pred_column)
        mse = mean_squared_error(y, pred_column)

        print(f"Column {i}: R-squared = {r2}, MSE = {mse}")
    print()

dataset = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/ResultsLogScaled.csv")
train_wo_dragon = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/MTrainUD_POC.csv")
test_wo_dragon = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/MTestUD_POC.csv")

combined = pd.concat([train_wo_dragon, test_wo_dragon], axis=0)['MoleFraction']
y = dataset.iloc[:, 6]
y_preds = dataset.iloc[:, 8:]

# Pitstop #1. Checking Metrics
metrics(y, y_preds)

# Reverse log scaling
for column in y_preds.columns:
        y_preds[column] = reverse_log_scaling(y_preds[column], combined)
y = reverse_log_scaling(y, combined)

# Pitstop #2. Checking Metrics
metrics(y, y_preds)

# Applying a transformation
y = 2*y
y_preds = y_preds*2

metrics(y, y_preds)
quit()