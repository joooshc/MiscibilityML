import pandas as pd
import time

def create_smi(smiles):
    # This part just generates smi files for the smiles codes
    smiles1 = smiles['SMILES1'].tolist()
    smiles2 = smiles['SMILES2'].tolist()

    with open('smiles1.smi', 'w') as smi_file:
        for smiles in smiles1:
            smi_file.write(str(smiles) + '\n')
    with open('smiles2.smi', 'w') as smi_file:
        for smiles in smiles2:
            smi_file.write(str(smiles) + '\n')

def create_csv():
    # This part builds a dataframe from the dragon descriptors in a text file
    with open("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/DragonDescriptors/dragon_descriptors.txt", 'r') as file:
        dragon_text = file.read()

    lines = dragon_text.strip().split("\n")
    descriptors = lines[0].split("\t")
    compounds = [dict(zip(descriptors, line.split("\t"))) for line in lines[1:]]
    dragon_df = pd.DataFrame(compounds)

    dragon_df.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/DragonDescriptors/dragon_descriptors.csv", index=False)

def create_dataset():
    smiles = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/master_unscaled.csv").iloc[:, 2:4]
    dragon_df = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/DragonDescriptors/dragon_descriptors.csv")

    # Create masks to check for equivalent smiles codes
    mask1 = smiles['SMILES1'].isin(dragon_df['NAME'])
    mask2 = smiles['SMILES2'].isin(dragon_df['NAME'])
    simultaneous_match = mask1 & mask2

    # Place the smiles codes that were found into separate single-column dataframes
    matched_smiles_df = smiles[simultaneous_match & mask1].copy()
    smiles1 = matched_smiles_df['SMILES1'].to_frame()
    smiles2 = matched_smiles_df['SMILES2'].to_frame()

    # Remove all duplicates to create a look up table of descriptors for unique compounds
    smiles1_unique = smiles1.drop_duplicates(subset=['SMILES1'])
    smiles2_unique = smiles2.drop_duplicates(subset=['SMILES2'])
    dragon_df_unique = dragon_df.drop_duplicates(subset=['NAME'])
    merged_smiles1_df = pd.merge(smiles1_unique, dragon_df_unique, left_on='SMILES1', right_on='NAME', how='left')
    merged_smiles2_df = pd.merge(smiles2_unique, dragon_df_unique, left_on='SMILES2', right_on='NAME', how='left')

    # Use the look-up table to populate the smiles entries (repeated or on their own) with their respective descriptors
    final_merged_smiles1 = pd.merge(smiles1, merged_smiles1_df, on='SMILES1', how='left')
    final_merged_smiles2 = pd.merge(smiles2, merged_smiles2_df, on='SMILES2', how='left')
    final_merged_smiles1.drop(columns=['NAME', 'No.'], inplace=True)
    final_merged_smiles2.drop(columns=['NAME', 'No.'], inplace=True)

    # Subtract all descriptors of smiles2 from smiles1
    subtracted_values = final_merged_smiles1.iloc[:, 1:].values - final_merged_smiles2.iloc[:, 1:].values
    subtracted_df = pd.DataFrame(subtracted_values, columns=final_merged_smiles1.columns[1:])
    result_df = pd.concat([smiles1.reset_index(drop=True), smiles2.reset_index(drop=True), subtracted_df], axis=1)

    result_df.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/DragonDescriptors/dragon_descriptors_pairs.csv", index=False)

def create_master():
    dragon_pairwise = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/DragonDescriptors/dragon_descriptors_pairs.csv")
    master_unscaled = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/master_unscaled.csv")

    filtered_master_unscaled = master_unscaled[master_unscaled.set_index(['SMILES1', 'SMILES2']).index.isin(dragon_pairwise.set_index(['SMILES1', 'SMILES2']).index)]
    filtered_master_unscaled = filtered_master_unscaled.reset_index(drop=True)
    dragon_pairwise_reset = dragon_pairwise.reset_index(drop=True)
    merged_df = pd.concat([filtered_master_unscaled, dragon_pairwise_reset], axis=1)

    print(merged_df)
    time.sleep(1000)

    merged_df.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/master_unscaled_dragon.csv", index=False)

create_master()