import numpy as np
import pubchempy as pcp
import json

class GetSMILES:
    
    def cas_to_dict(chemical_pairs):
        """
        Convert the CAS numbers from the chemical pairs into a dictionary where 
        each key is a unique CAS number and each value is the index of the CAS 
        number in the list of unique CAS numbers.

        Args:
        chemical_pairs (DataFrame): A DataFrame containing the chemical pairs.

        Returns:
        dict: A dictionary containing the CAS numbers as keys and their indices as values.
        """
        # Extract all CAS numbers from the DataFrame
        cas_numbers_1 = chemical_pairs['CAS1'].tolist()
        cas_numbers_2 = chemical_pairs['CAS2'].tolist()

        all_cas_numbers = cas_numbers_1 + cas_numbers_2
        unique_cas_numbers = set(all_cas_numbers)

        cas_dict = {cas_number: index for index, cas_number in enumerate(unique_cas_numbers)}

        return cas_dict

    @staticmethod
    def dict_to_smiles(cas_dict):
        """
        Get the SMILES codes for the compounds using the PubChem API and map them to the CAS numbers.

        Args:
        cas_dict (dict): A dictionary containing the CAS numbers as keys and their indices as values.

        Returns:
        dict: A dictionary containing the CAS numbers as keys and their corresponding SMILES codes as values.
        """
        count = 1
        print("Getting SMILES codes for dictionary, please wait...")
        cas_smiles_dict = {}
        for cas in cas_dict.keys():
            if not cas or cas.isspace(): 
                cas_smiles_dict[cas] = 'NaN'
            else:
                compounds = pcp.get_compounds(cas, 'name')
                if compounds:
                    cas_smiles_dict[cas] = compounds[0].canonical_smiles
                else:
                    cas_smiles_dict[cas] = 'NaN'
                    
            count += 1
            print(f"Progress: {np.round(100*(count/len(cas_dict)), 2)}%")
        
        # Save CAS-SMILES dictionary to a JSON file
        with open('cas_smiles_dict.json', 'w') as file:
            json.dump(cas_smiles_dict, file)
        
        print(f"SMILES codes obtained for dictionary and saved to cas_smiles_dict.json")