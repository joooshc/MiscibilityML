import pandas as pd
import json

def replacecas(folder):
    with open(f"Datasets/{folder}/cas_smiles_dict.json", "r") as f:
        cas_smiles_dict = json.load(f)

    iupac = pd.read_csv(f"Datasets/{folder}/diff_dataset_raw.csv")
    CAS1 = iupac["CAS1"].values
    CAS2 = iupac["CAS2"].values
    SMILES1 = []; SMILES2 = []

    for i in range(len(CAS1)):
        SMILES1.append(cas_smiles_dict[CAS1[i]])
        SMILES2.append(cas_smiles_dict[CAS2[i]])

    iupac.drop(columns = ["CAS1", "CAS2"], inplace = True)
    iupac.insert(2, "SMILES1",  SMILES1)
    iupac.insert(3, "SMILES2",  SMILES2)
    iupac.to_csv(f"Datasets/{folder}/diff_raw_smiles.csv", index = False)

replacecas("IUPAC")
replacecas("DDBprocesseddata/x1")

df1 = pd.read_csv("Datasets/IUPAC/diff_raw_smiles.csv")
df2 = pd.read_csv("Datasets/DDBprocesseddata/x1/diff_raw_smiles.csv")
df3 = pd.read_csv("Datasets/AQSolDB/diff_df.csv")
df4 = pd.concat([df1, df2, df3], ignore_index = True)
df4.to_csv("Datasets/master_unscaled.csv", index = False)