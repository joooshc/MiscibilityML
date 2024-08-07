import pandas as pd
from scaler import outlier_removal

def cas_list(df):
    cas = []
    cas1 = df["SMILES1"].tolist()
    cas1 = [x.replace(" ", "") for x in cas1]
    cas2 = df["SMILES2"].tolist()
    cas2 = [x.replace(" ", "") for x in cas2]
    for i in range(len(cas1)):
        cas.append(f"{cas1[i]}, {cas2[i]}")

    return cas

def filter(df, common_pairs):
    new_df = df.copy()
    cas1 = df["SMILES1"].tolist()
    cas2 = df["SMILES2"].tolist()

    for i in range(len(cas1)):
        pair = f"{cas1[i]}, {cas2[i]}"
        if pair in common_pairs:
            new_df.drop(i, inplace=True)

    val_df = pd.concat([df, new_df]).drop_duplicates(keep=False)

    return new_df, val_df
        
def main():
    master = pd.read_csv("Datasets/master_log_outliers_inc.csv")
    SigAl = pd.read_csv("Datasets/Sigma-Aldrich/Sigma-Aldrich_data.csv")
    master = master.drop_duplicates()
    SigAl = SigAl.drop_duplicates()
    master.reset_index(drop=True, inplace=True)
    SigAl.reset_index(drop=True, inplace=True)

    SigAl_cas = cas_list(SigAl)
    master_cas = cas_list(master)

    common_pairs = [element for element in master_cas if element in SigAl_cas]

    filtered_df, val_df = filter(master, common_pairs)
    # outlier_removal(filtered_df)
    # outlier_removal(val_df)
    print(f"Orignal df shape: {master.shape}, Filtered df shape: {filtered_df.shape}. Validation df shape: {val_df.shape}")

    return filtered_df, val_df

filtered_df, validation_df = main()
filtered_df.to_csv("Datasets/master_log_trainaaaaaa.csv", index=False)
validation_df.to_csv("Datasets/master_log_testaaaaaa.csv", index=False)