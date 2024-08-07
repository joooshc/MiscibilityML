import os
import json
import pandas as pd
from JSON_to_CSV import JSON_to_CSV
from GetSMILES import GetSMILES
from ExtractFeatures import ExtractFeatures

class CASManager:
    def __init__(self, root_directory):
        self.datasets_path = os.path.join(root_directory, "Datasets", "IUPAC")
        self.cas_smiles_file = os.path.join(self.datasets_path, 'cas_smiles_dict.json')
        self.cas_pubchem_props_file = os.path.join(self.datasets_path, 'cas_pubchem_props_dict.json')
        self.cas_rdkit_descriptors_file = os.path.join(self.datasets_path, 'cas_rdkit_descriptors_dict.json')
    
    def smiles_gen(self, iupac_dataset_path):
        # Generating a dataframe from the JSON file
        jcsv = JSON_to_CSV()
        jcsv.load_data(iupac_dataset_path)
        df = jcsv.process_data()

        # Check if the json file exists, if it doesn't then generate it
        if not os.path.isfile(self.cas_smiles_file):
            cas_dict = GetSMILES.cas_to_dict(df)
            GetSMILES.dict_to_smiles(cas_dict)
        else:
            with open(self.cas_smiles_file, 'r') as f:
                cas_dict = json.load(f)
                
        return cas_dict, df
    
    def pubchem_props_gen(self, cas_dict):
        extF = ExtractFeatures()
        # Check if the json file with pubchem properties exists, if it doesn't then generate it
        if not os.path.isfile(self.cas_pubchem_props_file):
            cas_pubchem_props_dict = extF.fetch_pubchem_props(cas_dict, self.cas_pubchem_props_file.split('.')[0])
        else:
            with open(self.cas_pubchem_props_file, 'r') as f:
                cas_pubchem_props_dict = json.load(f)
        
        return cas_pubchem_props_dict
    
    def rdkit_descriptors_gen(self, cas_dict):
        extF = ExtractFeatures()
        # Check if the json file with RDKit descriptors exists, if it doesn't then generate it
        if not os.path.isfile(self.cas_rdkit_descriptors_file):
            cas_rdkit_descriptors_dict = extF.rdkit_descriptors_dict(cas_dict)
            with open(self.cas_rdkit_descriptors_file, 'w') as f:
                json.dump(cas_rdkit_descriptors_dict, f)
        else:
            with open(self.cas_rdkit_descriptors_file, 'r') as f:
                cas_rdkit_descriptors_dict = json.load(f)
        
        return cas_rdkit_descriptors_dict
    
    def merge_dicts_to_df(self, df, cas_pubchem_props_dict, cas_rdkit_descriptors_dict):
        data1, data2 = [], []
        data_diff = []
        for index, row in df.iterrows():
            cas1_data = {
                **cas_pubchem_props_dict[row['CAS1']],
                **cas_rdkit_descriptors_dict[row['CAS1']]
            }
            cas2_data = {
                **cas_pubchem_props_dict[row['CAS2']],
                **cas_rdkit_descriptors_dict[row['CAS2']]
            }
            data1.append(cas1_data)
            data2.append(cas2_data)

            # Create a copy of the cas1_data dictionary
            cas1_data_diff = dict(cas1_data)

            # Remove the elements that aren't properties or descriptors
            keys_to_delete = list(cas1_data_diff.keys())[:2]
            for key in keys_to_delete:
                del cas1_data_diff[key]

            # Now compute the diff_data with the cas1_data_diff dictionary
            cas1_data_diff = {k: float(v) for k, v in cas1_data_diff.items()}
            diff_data = {}
            for key in cas1_data_diff:
                if key in cas2_data:
                    try:
                        diff_data[key] = cas1_data_diff[key] - float(cas2_data[key])
                    except ValueError:
                        print(f'Error: Cannot convert {cas2_data[key]} to float')
                else:
                    pass

            data_diff.append(diff_data)

        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        df_diff = pd.DataFrame(data_diff)

        print(f"The new datasets shapes are: {df1.shape, df2.shape, df_diff.shape}")

        df_first_six = df.iloc[:, :6]
        df_diff = pd.concat([df_first_six.reset_index(drop=True), df_diff], axis=1)
        df1 = pd.concat([df_first_six.reset_index(drop=True), df1], axis=1)
        df2 = pd.concat([df_first_six.reset_index(drop=True), df2], axis=1)

        df1_save_path = os.path.join(self.datasets_path, 'compound1_dataset_raw.csv')
        df2_save_path = os.path.join(self.datasets_path, 'compound2_dataset_raw.csv')
        df_diff_save_path = os.path.join(self.datasets_path, 'diff_dataset_raw.csv')
        
        df1.to_csv(df1_save_path, index=False)
        df2.to_csv(df2_save_path, index=False)
        df_diff.to_csv(df_diff_save_path, index=False)

        return df1, df2, df_diff


