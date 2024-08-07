import pandas as pd
import numpy as np

dataset = pd.read_csv("Datasets/TrainTestData/feature_rankings.csv")
dataset["Loss"] = np.round(dataset["Loss"], 2)
agg_df = dataset.groupby("Loss").agg(Features=("FeatureName", "unique"))
agg_df.sort_values(by="Loss", ascending=False, inplace=True)
agg_df.reset_index(inplace=True)
print(agg_df.head())
# agg_df.to_csv("Datasets/TrainTestData/feature_rankings_processed.csv", index=False)

feature_list = []
nested_list = agg_df["Features"].tolist()
for i in range(len(nested_list)):
    feature_list.extend(nested_list[i])

with open("Datasets/TrainTestData/feature_rankings_processed.txt", "w") as file:
    file.write(f"{feature_list}")