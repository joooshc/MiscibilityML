import json, os, time
import pandas as pd
import numpy as np

class JSON_to_CSV:
    # def __init__(self):
    #     # Define directories for data loading
    #     self.current_directory = os.getcwd()
    #     self.parent_directory = os.path.dirname(self.current_directory)
    #     self.main_directory = os.path.join(self.parent_directory)
    #     self.file_directory = os.path.join(self.main_directory, "Datasets")
    #     os.chdir(self.file_directory)

    def load_data(self, filename):
        # Load data from a JSON file
        with open("Datasets/DDBscrapeddata/IUPAC_vap_output.json") as json_file:
            self.dataset = json.load(json_file)

    def process_data(self):
        # Initialize lists to store data
        compound_1 = []
        compound_2 = []
        mole_fractions = []
        corresp_temps = []
        cas_nums_1 = []
        cas_nums_2 = []

        # Iterate over each pair in the dataset
        for pairs, values in self.dataset.items():
            # Split the pair into separate compounds
            compounds = pairs.split(' & ')
            mole_fraction_list = values['mole fraction:']
            corresp_temps_list = values['corresp temps (C):']
            cas_nums = values['cas num:']
            
            # Split CAS numbers if they are a string
            if isinstance(cas_nums, str):
                cas_nums = cas_nums.split(', ')
            elif isinstance(cas_nums, list) and len(cas_nums) == 1:
                cas_nums = cas_nums[0].split(', ')
            
            # Assign CAS numbers
            cas_num_1 = cas_nums[0]
            cas_num_2 = cas_nums[1] if len(cas_nums) > 1 else None
            
            # Iterate over each mole fraction and corresponding temperature
            for mole_fraction, temp in zip(mole_fraction_list, corresp_temps_list):
                # Append all values to their respective lists
                compound_1.append(compounds[0])
                compound_2.append(compounds[1])
                mole_fractions.append(mole_fraction)
                corresp_temps.append(temp)
                cas_nums_1.append(cas_num_1)
                cas_nums_2.append(cas_num_2)

        # Build a DataFrame from the processed lists
        dataframe = {
            'Compound1': compound_1,
            'Compound2': compound_2,
            'CAS1': cas_nums_1,
            'CAS2': cas_nums_2,
            'MoleFraction': mole_fractions,
            'Temperature': corresp_temps,
        }

        df = pd.DataFrame(dataframe)
        # Save the DataFrame to a CSV file
        df.to_csv('ddb_dataset_combined.csv', index=False)
        print(f"The dataframe's shape is: {df.shape}")

        return df
