import pandas as pd
import numpy as np
from tqdm import tqdm
import numpy as np
import json

def prop_dict(compounds, logS, nWater):
    mol_fracs = []
    for i in range(len(compounds)):
        x = np.power(10, logS[i]) / (np.power(10, logS[i]) + nWater)
        mol_fracs.append(x)
    return mol_fracs

def iupac_formatter(compounds, smiles, mol_fracs):
    output = {}
    for i in tqdm(range(len(compounds))):
        names = f"{compounds[i]} & water"

        subdict = {"mole fraction" : mol_fracs[i],
                "corresp temps (C)" : [25],
                "smiles" : [smiles[i], "O"]}
        iupac_dict = {names : subdict}
        output.update(iupac_dict)
    return output

def combined_df(compounds, smiles, mol_fracs):
    watersmiles = ["O"] * len(compounds)
    water = ["water"] * len(compounds)
    df = pd.DataFrame(zip(compounds, water, smiles, watersmiles, mol_fracs), columns=["Compound1", "Compound2", "SMILES1", "SMILES2", "MoleFraction"])
    return df

def main():
    dataset = pd.read_csv("Datasets/AQSolDB/curated-solubility-dataset.csv")
    dataset.dropna(inplace=True)
    rows = dataset.shape[0]
    print(rows)

    compound1 = dataset["Name"].to_list()
    logS = dataset["Solubility"].tolist()
    nWater = float(1000/18.01528)
    smiles = dataset["SMILES"].tolist()

    mol_fracs = prop_dict(compound1, logS, nWater)
    iupac_dict = iupac_formatter(compound1, smiles, mol_fracs)

    df = combined_df(compound1, smiles, mol_fracs)
    df.to_csv("Datasets/AQSolDB/combined.csv", index=False)
    # with open(f"Datasets/AQSolDB/iupac_output.json", "w") as outfile:
    #     json.dump(iupac_dict, outfile)

    print(len(iupac_dict))

main()