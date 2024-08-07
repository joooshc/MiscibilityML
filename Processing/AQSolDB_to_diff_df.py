import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def json_combiner(filename):
    output = {}
    for i in tqdm(range(0, 48)):
        with open(f"Datasets/AQSolDB/{filename}{i}.json", "r") as f:
            data = json.load(f)
        output.update(data)
    
    with open(f"Datasets/AQSolDB/{filename}.json", "w") as f:
        json.dump(output, f)
    return output

# pubchem = json_combiner("pubchem_properties")
# rdkit = json_combiner("rdkit_descriptors")

def table_to_dict(table): #Converts the tables to dictionaries
    dict = {}
    cols = table.columns.tolist()

    for i in range(len(cols)):
        data = table[cols[i]].values.tolist()
        dict.update({cols[i]: data})

    return dict

with open("Datasets/AQSolDB/pubchem_properties.json", "r") as f:
    pubchem_props = json.load(f)

with open("Datasets/AQSolDB/rdkit_descriptors.json", "r") as f:
    rdkit_desc = json.load(f)

with open("Datasets/AQSolDB/pubchem_properties0.json", "r") as f:
    water_pc = json.load(f)

with open("Datasets/AQSolDB/rdkit_descriptors0.json", "r") as f:
    water_rd = json.load(f)

with open("Datasets/AQSolDB/iupac_output.json", "r") as f:
    iupac = json.load(f)

def diff_creator():
    diff_dict = {}

    for key in raw_dict:
        try:
            raw_dict[key] = [float(i) for i in raw_dict[key]]
            raw_water[key] = [float(i) for i in raw_water[key]]
        except:
            print(key)
            raw_dict[key] = [float(i[0]) for i in raw_dict[key]]
            raw_water[key] = [float(i[0]) for i in raw_water[key]]
            diff_dict[key] = (np.array(raw_dict[key]) - np.array(raw_water[key])).tolist()
            continue
        diff_dict[key] = (np.array(raw_dict[key]) - np.array(raw_water[key])).tolist()

    diff_df = pd.DataFrame.from_dict(diff_dict)
    print(diff_dict["HeavyAtomCount"])
    return diff_df

pubchem_df = pd.DataFrame.from_dict(pubchem_props, orient="index")
rdkit_df = pd.DataFrame.from_dict(rdkit_desc, orient="index")
water_pc = pd.DataFrame.from_dict(water_pc, orient="index")
water_rd = pd.DataFrame.from_dict(water_rd, orient="index")

water_df = pd.concat([water_pc, water_rd], axis=1)
raw_df = pd.concat([pubchem_df, rdkit_df], axis=1)
raw_df.dropna(inplace=True)

raw_dict = table_to_dict(raw_df.iloc[:, 2:])
raw_water = table_to_dict(water_df.iloc[:, 2:])

diff_df = diff_creator()
print(diff_df.shape)

compounds = raw_df.index.tolist()
compound1 = [i.split(" & ")[0] for i in compounds]
compound2 = [i.split(" & ")[1] for i in compounds]

smiles = []
wsmile = []
mol_frac = []
temp = []

for compound in compounds:
    subdict = iupac[compound]
    smiles.append(subdict["smiles"][0])
    wsmile.append("O")
    mol_frac.append(subdict["mole fraction"])
    temp.append(25)

metadata_df = pd.DataFrame(zip(compound1, compound2, smiles, wsmile, mol_frac, temp), columns=["Compound1", "Compound2", "SMILES1", "SMILES2",  "MoleFraction", "Temperature"])
diff_df = pd.concat([metadata_df, diff_df], axis=1)
print(diff_df)

diff_df.to_csv("Datasets/AQSolDB/diff_df.csv", index=False)