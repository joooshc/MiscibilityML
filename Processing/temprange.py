import pandas as pd

def insert_pairs(dataset):
    smiles1 = dataset["SMILES1"].tolist()
    smiles2 = dataset["SMILES2"].tolist()

    pair_list = []
    for i in range(dataset.shape[0]):
        pair_list.append(f'{smiles1[i]} + {smiles2[i]}')
    
    dataset.insert(0, "Pair", pair_list)

    return dataset

def minmax_calc(dataset):
    dataset = insert_pairs(dataset)

    dataset.drop_duplicates(inplace=True)
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    output = dataset.groupby(["SMILES1", "SMILES2"]).agg({"Temperature": ["min", "max"]}).reset_index()
    output.columns = ["SMILES1", "SMILES2", "MinTemp", "MaxTemp"]

    data_dict = {}
    output = insert_pairs(output)

    for i in range(output.shape[0]):
        key = f"{output['SMILES1'][i]} + {output['SMILES2'][i]}"
        data_dict[key] = [output["MinTemp"][i], output["MaxTemp"][i]]

    return data_dict

def minmax_insert(data_dict, dataset):
    if "Pair" not in dataset.columns:
        dataset = insert_pairs(dataset)
    min_list = []
    max_list = []

    for i in range(dataset.shape[0]):
        pair = dataset["Pair"][i]
        min_list.append(data_dict[pair][0])
        max_list.append(data_dict[pair][1])

    dataset.insert(5, "MinTemp", min_list)
    dataset.insert(6, "MaxTemp", max_list)

    del dataset["Pair"]

    return dataset

# dataset = pd.read_csv("Datasets/master_unscaled_dragon.csv")
# print(dataset.shape)
# data_dict = minmax_calc(dataset)
# dataset = minmax_insert(data_dict, dataset)
# print(dataset.shape)
# print(dataset.head())