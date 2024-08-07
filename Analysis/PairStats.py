import pandas as pd
import numpy as np

train_u = pd.read_csv("Results/Stats/ValidTrain.csv")
test_u = pd.read_csv("Results/Stats/AllResults_Unique-stats.csv")
df = pd.concat([train_u, test_u], ignore_index=True)
df.drop_duplicates(inplace=True)

train = pd.read_csv("Datasets/TrainTestData/MTrainUD_POC.csv")
test = pd.read_csv("Datasets/TrainTestData/MTestUD_POC.csv")
# df = pd.concat([train, test], ignore_index=True)
# df.drop_duplicates(inplace=True)

# print(df_u.shape)
# print(df.shape)


mf_nested = test_u["true_mfs"].tolist()
mf_nested = mf_nested + train_u["MoleFractions"].tolist()
mf_nested = [m.split(", ") for m in mf_nested]

lengths = []
x = 0
for m in mf_nested:

    lengths.append(len(m))

    if len(m) > 100:
        print(df.iloc[x, :])
    x += 1

print(np.max(lengths))
print(np.min(lengths))