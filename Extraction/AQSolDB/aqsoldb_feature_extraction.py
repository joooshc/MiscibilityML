import pandas as pd
import numpy as np
from tqdm import tqdm
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors
import json

def fetch_pubchem_props(cas_smiles_dict, file_name, i):
    # List of properties to retrieve
    ToRetrieve = ["MolecularFormula", "MolecularWeight", "XLogP", "ExactMass", 
                "MonoisotopicMass", "TPSA", "Complexity", "HBondDonorCount", 
                "HBondAcceptorCount", "RotatableBondCount", "HeavyAtomCount", 
                "Volume3D", "XStericQuadrupole3D", "YStericQuadrupole3D", 
                "ZStericQuadrupole3D", "FeatureCount3D", "FeatureAcceptorCount3D", 
                "FeatureDonorCount3D", "FeatureAnionCount3D", "FeatureRingCount3D", 
                "FeatureHydrophobeCount3D", "ConformerModelRMSD3D", "EffectiveRotorCount3D", 
                "ConformerCount3D"]

    # Loop to get properties with smiles codes. Also checks for problematic compounds
    property_dict = {}
    for cas, smiles in tqdm(cas_smiles_dict.items(), desc="Getting PubChem properties"):
        try:
            RawProperty = pcp.get_properties(ToRetrieve, smiles, "smiles") 
            PropertyDict = RawProperty[0]
        except pcp.BadRequestError:
            PropertyDict = {prop: 'NaN' for prop in ToRetrieve}
        
        property_dict[cas] = PropertyDict

    # Save the property dictionary as a json file
    with open(f'{file_name}{i}.json', 'w') as f:
        json.dump(property_dict, f)

    return property_dict

def rdkit_descriptors_dict(cas_dict, file_name, i):
    descriptor_dict = {}
    for cas_num, smiles in cas_dict.items():
        compound_descriptors = {}
        try:
            compound = Chem.MolFromSmiles(smiles)
            if compound is None:
                raise ValueError('Unable to create compound from SMILES string')
            for descriptor_name, descriptor_fn in Descriptors.descList:
                descriptor_value = descriptor_fn(compound)
                compound_descriptors[descriptor_name] = descriptor_value
        except (ValueError, AttributeError) as e:
            # Handle any compounds that can't be created or descriptors that can't be computed
            for descriptor_name, _ in Descriptors.descList:
                compound_descriptors[descriptor_name] = np.nan
        descriptor_dict[cas_num] = compound_descriptors
    
    with open(f'{file_name}{i}.json', 'w') as f:
        json.dump(descriptor_dict, f)
    return descriptor_dict

# with open("Datasets/AQSolDB/iupac_output.json", "r") as file:
#     data_dict = json.load(file)

# keylist = list(data_dict.keys())
# # for key in data_dict.keys():
# #     cas_smiles_dict[key] = data_dict[key]["smiles"][0]

# x = 0
# for j in range(50):
#     cas_smiles_dict = {}
#     for i in range(x, x + 200):
#         cas_smiles_dict[keylist[x]] = data_dict[keylist[x]]["smiles"][0]
#         x += 1
for j in range(0, 1):
    cas_smiles_dict = {"water" : "O"}
    property_dict = fetch_pubchem_props(cas_smiles_dict, "Datasets/AQSolDB/pubchem_properties", j)
    descriptor_dict = rdkit_descriptors_dict(cas_smiles_dict, "Datasets/AQSolDB/rdkit_descriptors", j)