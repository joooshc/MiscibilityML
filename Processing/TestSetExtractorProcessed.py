import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_quartile_hists(first_quartile, last_quartile):
    plt.hist(first_quartile, bins=20, edgecolor='black')
    plt.title('Histogram of 1st Quartile (25%)')
    plt.xlabel('MoleFraction')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(last_quartile, bins=20, edgecolor='black')
    plt.title('Histogram of Last Quartile (75%)')
    plt.xlabel('MoleFraction')
    plt.ylabel('Frequency')
    plt.show()

def split_quartiles(y, X, metadata):
    first_quartile_value = y.quantile(0.25)
    third_quartile_value = y.quantile(0.75)

    first_quartile_mask = y <= first_quartile_value
    last_quartile_mask = y >= third_quartile_value
    interquartile_mask = (y > first_quartile_value) & (y < third_quartile_value)

    first_quartile_y = y[first_quartile_mask]
    last_quartile_y = y[last_quartile_mask]
    interquartile_y = y[interquartile_mask]

    first_quartile_X = X[first_quartile_mask]
    last_quartile_X = X[last_quartile_mask]
    interquartile_X = X[interquartile_mask]

    first_quartile_meta = metadata[first_quartile_mask]
    last_quartile_meta = metadata[last_quartile_mask]
    interquartile_meta = metadata[interquartile_mask]

    return (first_quartile_meta, first_quartile_y, first_quartile_X), (interquartile_meta, interquartile_y, interquartile_X), (last_quartile_meta, last_quartile_y, last_quartile_X)

def unique_mole_fractions(dataset):
    # Group by the SMILES codes and count the number of occurrences for each unique compound
    master_smiles = dataset.iloc[:, 2:4]
    grouped = dataset.groupby([master_smiles.columns[0], master_smiles.columns[1]]).size().reset_index(name='count')
    repeated_pairs = grouped[grouped['count'] > 0]

    # Calculate and remove 20% of the total repeated pairs
    num_to_remove = int(0.2 * len(repeated_pairs))
    test_set = repeated_pairs.sample(num_to_remove)
    training_set = repeated_pairs.drop(test_set.index)

    # Select rows from master_unscaled with equivalent SMILES1 and SMILES2 in training_set
    training_data = pd.merge(dataset, training_set[[master_smiles.columns[0], master_smiles.columns[1]]], 
                            on=[master_smiles.columns[0], master_smiles.columns[1]], 
                            how='inner')
    test_data = pd.merge(dataset, test_set[[master_smiles.columns[0], master_smiles.columns[1]]], 
                        on=[master_smiles.columns[0], master_smiles.columns[1]], 
                        how='inner')
    
    return training_data, test_data

name = 'QuantileMinMax'

dataset = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragon/{name}Train.csv")
test_dataset = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragon/{name}Test.csv")
combined_dataset = pd.concat([dataset, test_dataset], axis=0)

y_train = dataset['MoleFraction']
X_train = dataset.iloc[:, 7:]
metadata = dataset.iloc[:, :6]
X_test = test_dataset.iloc[:, 7:]
y_test = test_dataset['MoleFraction']
metadata_test = test_dataset.iloc[:, :6]

combined_X = pd.concat([X_train, X_test], axis=0)
combined_y = pd.concat([y_train, y_test], axis=0)

(train_first_quartile_meta, train_first_quartile_y, train_first_quartile_X), (train_interquartile_meta, train_interquartile_y, train_interquartile_X), (train_last_quartile_meta, train_last_quartile_y, train_last_quartile_X) = split_quartiles(y_train, X_train, metadata)
(test_first_quartile_meta, test_first_quartile_y, test_first_quartile_X), (test_interquartile_meta, test_interquartile_y, test_interquartile_X), (test_last_quartile_meta, test_last_quartile_y, test_last_quartile_X) = split_quartiles(y_test, X_test, metadata_test)

train_1st_quartile = pd.concat([train_first_quartile_meta, train_first_quartile_y, train_first_quartile_X], axis=1)
train_IQ_quartile = pd.concat([train_interquartile_meta, train_interquartile_y, train_interquartile_X], axis=1)
train_last_quartile = pd.concat([train_last_quartile_meta, train_last_quartile_y, train_last_quartile_X], axis=1)
test_1st_quartile = pd.concat([test_first_quartile_meta, test_first_quartile_y, test_first_quartile_X], axis=1)
test_IQ_quartile = pd.concat([test_interquartile_meta, test_interquartile_y, test_interquartile_X], axis=1)
test_last_quartile = pd.concat([test_last_quartile_meta, test_last_quartile_y, test_last_quartile_X], axis=1)

print(f"train 1st Q: {train_1st_quartile.shape}, test 1st Q: {test_1st_quartile.shape}")
print(f"train IQ: {train_IQ_quartile.shape}, test IQ: {test_IQ_quartile.shape}")
print(f"train last Q: {train_last_quartile.shape}, test last Q: {test_last_quartile.shape}")

count = (train_1st_quartile['Compound2'] == 'Water').sum()

train_2Q = pd.concat([train_IQ_quartile, train_last_quartile], axis=0)
test_2Q = pd.concat([test_IQ_quartile, test_last_quartile], axis=0)

train_2Q.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quartiles/{name}Train_2Q.csv", index=False)
test_2Q.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quartiles/{name}Test_2Q.csv", index=False)

# train_1st_quartile.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quartiles/{name}Train_Q1.csv", index=False)
# train_IQ_quartile.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quartiles/{name}Train_IQ.csv", index=False)
# train_last_quartile.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quartiles/{name}Train_LQ.csv", index=False)
# test_1st_quartile.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quartiles/{name}Test_Q1.csv", index=False)
# test_IQ_quartile.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quartiles/{name}Test_IQ.csv", index=False)
# test_last_quartile.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quartiles/{name}Test_LQ.csv", index=False)