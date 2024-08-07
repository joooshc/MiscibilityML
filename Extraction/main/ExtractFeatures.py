import json
import pandas as pd
import numpy as np
import pubchempy as pcp
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

class ExtractFeatures:

    def fetch_pubchem_props(self, cas_smiles_dict, file_name):
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
        with open(f'{file_name}.json', 'w') as f:
            json.dump(property_dict, f)

        return property_dict

    def rdkit_descriptors_dict(self, cas_dict):
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
        
        return descriptor_dict
        