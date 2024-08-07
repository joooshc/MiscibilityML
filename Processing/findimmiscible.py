import pandas as pd
import time as t

df = pd.read_csv("Datasets/TrainTestData/MTrainUD_POC.csv")
test = pd.read_csv("Datasets/TrainTestData/MTestUD_POC.csv")
print(df.shape)
print(test.shape)

def finder(df):
    smiles1 = df["SMILES1"].values.tolist()
    smiles2 = df["SMILES2"].values.tolist()

    tofind = ["O", "C1=CC=CC=C1"]

    indexes = []
    for i in range(len(smiles1)):
        if smiles1[i] in tofind and smiles2[i] in tofind:
            indexes.append(i)
    print(indexes)
    return indexes

indexes = finder(df)

for i in indexes:
    new = pd.DataFrame(df.iloc[i]).T
    print(new)
    test = pd.concat([test, new])

for i in indexes:
    df.drop(i, inplace=True)

df.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

finder(test)
print(df.shape)
print(test.shape)
df.to_csv("Datasets/TrainTestData/Train2.csv", index=False)
test.to_csv("Datasets/TrainTestData/Test2.csv", index=False)