import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import numpy as np

def get_fg(smiles):
    df = pd.DataFrame()
    
    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        property_vals = []
        for com in commands:
            val = getattr(Chem.Fragments, com)(mol)
            property_vals.append(val)
        new_df = pd.DataFrame.from_dict({smile: property_vals}, orient="index", columns=commands)
        df = pd.concat([df, new_df])
    df.reset_index(inplace=True)

    cols = df.iloc[:, 1:].columns.tolist()
    sums = ["SMILE"]
    nonzero = ["SMILE"]
    count_dict = {}

    for col in cols:
        if sum(df[col]) == 0:
            del df[col]
        else:
            count = np.count_nonzero(df[col])
            percent = np.round((count/len(df[col]))*100,2)
            nonzero.append(count)
            sums.append(sum(df[col]))
            count_dict[col] = [count, sum(df[col]), percent]

    new_rows = pd.DataFrame([sums, nonzero], index=["Total", "Count"], columns=df.columns.tolist())
    df = pd.concat([df, new_rows], axis=0)
    count_df = pd.DataFrame.from_dict(count_dict, orient="index", columns=["Count", "Total", "Percentage"])

    return df, count_df

def tidy_count_df(df):
    df.reset_index(inplace=True)
    df.columns = ["Index", "Count", "Total", "Percentage"]
    df.sort_values(by="Count", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    indexcol = df["Index"].tolist()
    definitions = []
    for col in indexcol:
        definitions.append(getattr(Chem.Fragments, col).__doc__)
    df.insert(1, "Definition", definitions)
    return df

if __name__ == "__main__":
    SAVE = True
    name = "ValidTrain"
    data = pd.read_csv("Results/Stats/ValidTrain.csv")
    # data = pd.read_csv("Datasets/TrainTestData/NoDragon/LogTrain.csv")

    smiles1 = data["SMILES1"].tolist()
    smiles2 = data["SMILES2"].tolist()
    compounds1 = data["Compound1"].tolist()
    compounds2 = data["Compound2"].tolist()

    commands = [command for command in dir(Chem.Fragments) if command.startswith("fr_")]

    smiles = smiles1 + smiles2
    df1, cdf1 = get_fg(smiles)
    cdf1 = tidy_count_df(cdf1)

    # df1, cdf1 = get_fg(smiles1)
    # df2, cdf2 = get_fg(smiles2)

    # df1.insert(0, "Compound1", compounds1+["Total", "Count"])
    # df2.insert(0, "Compound2", compounds2+["Total", "Count"])

    # df1cols = df1.columns.tolist()
    # df2cols = df2.columns.tolist()

    # df3 = pd.concat([df1, df2], axis=1, ignore_index=True)
    # df3.columns = df1cols + df2cols
    # print(df3.head(), df3.shape)

    # cdf1 = tidy_count_df(cdf1)
    # cdf2 = tidy_count_df(cdf2)

    if SAVE:
        # df3.to_csv(f"Results/Stats/FunctionalGroup/{name}.csv", index=False)
        cdf1.to_csv(f"Results/Stats/FunctionalGroup/{name}Stats.csv", index=False)
        df1.to_csv(f"Results/Stats/FunctionalGroup/{name}.csv", index=False)
        # cdf2.to_csv(f"Results/Stats/FunctionalGroup/{name}StatsC2.csv", index=False)