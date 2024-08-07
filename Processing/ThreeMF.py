import pandas as pd
import numpy as np
from tqdm import tqdm

def get_indices(lst):
    min_val, max_val = min(lst), max(lst)
    median_val = np.median(lst)
    min_idx = lst.index(min_val)
    max_idx = lst.index(max_val)
    median_idx = lst.index(sorted(lst)[len(lst)//2])
    return min_idx, median_idx, max_idx

def sort_dataframe(dataset, column):
    agg_df = dataset.groupby(['Compound1', 'Compound2', 'SMILES1', 'SMILES2']).agg(lambda x: list(x)).reset_index()
    agg_df['Temp_Length'] = agg_df['Temperature'].apply(len)
    agg_df = agg_df.sort_values(by='Temp_Length', ascending=False).reset_index(drop=True)
    
    # Filtering rows where the length of the Temperature list is at least 5
    compounds_to_plot = agg_df[agg_df['Temp_Length'] >= 5].reset_index(drop=True)
    compounds_to_plot.drop(columns=['Temp_Length', 'MW', 'AMW', 'Sv'], inplace=True)

    # Extract min, median, and max indices from 'MoleFraction' and apply to all other columns
    def apply_indices(row):
        min_idx, median_idx, max_idx = get_indices(row['MoleFraction'])
        for col in row.index[6:]:
            row[col] = [row[col][min_idx], row[col][median_idx], row[col][max_idx]]
        return row

    compounds_to_plot = compounds_to_plot.apply(apply_indices, axis=1)
    
    return compounds_to_plot

def elongate_dataframe(df):
    elongated_rows = []
    
    for _, row in df.iterrows():
        for i in range(3):  # since each list has 3 elements: min, median, max
            new_row = {}
            for col in df.columns:
                if isinstance(row[col], list):
                    new_row[col] = row[col][i]
                else:
                    new_row[col] = row[col]
            elongated_rows.append(new_row)

    return pd.DataFrame(elongated_rows)

file_name = ['Log', 'Quantile', 'MinMax', 'QuantileMinMax', 'LogMinMax'] #, 'None']

for fname in tqdm(file_name):
    train_dataset = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragon/{fname}Train.csv")
    test_dataset = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragon/{fname}Test.csv")
    
    print("Processing:", fname)
    
    # Sorting and filtering the dataset
    compounds_train = sort_dataframe(train_dataset, fname)
    compounds_test = sort_dataframe(test_dataset, fname)
    
    # Elongating the DataFrame
    elongated_compounds_train = elongate_dataframe(compounds_train)
    elongated_compounds_test = elongate_dataframe(compounds_test)
    
    # Save to csv
    elongated_compounds_train.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/ThreeMF/NoDragon/{fname}Train.csv", index=False)
    elongated_compounds_test.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/ThreeMF/NoDragon/{fname}Test.csv", index=False)

    

