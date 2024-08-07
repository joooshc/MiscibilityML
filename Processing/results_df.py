import pandas as pd
from temprange import *
from scaler import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import re

def import_results(model, filename):
    with open(f"Results/LightGBM/Immiscible/{filename}.txt", "r") as f:
        y_pred = f.read().split(",")
    y_pred = [x.strip() for x in y_pred]
    del y_pred[-1]
    return y_pred

def unscaler(col):
    for i in range(len(col)):
        col[i] = np.exp(float(col[i]))
        if col[i] <= 1e-7:
            col[i] = 0
    return col

def pred_df_creator():
    dataset = pd.read_csv("Datasets/TrainTestData/ImmiscibleSets/NoneTest.csv")
    scalers = ["Log", "MinMax", "Quantile", "QuantileMinMax", "LogMinMax"]
    df = dataset.iloc[:, 0:8]
    models = ["dart", "gbdt", "rf"]

    mole_fracs = df["MoleFraction"].tolist()
    mole_fracs = np.log(mole_fracs)
    df.insert(8, "LogMoleFraction", mole_fracs)
    
    for model in models:
        for scaler in scalers:
            y_pred = import_results(model, f"{scaler}-{model}")
            df.insert(df.shape[1], f"{model}-{scaler}", y_pred)

    df.to_csv(f"Results/Immiscible.csv", index = False)
    print(df.head())
    return df

pred_df_creator()
quit()
def unscale_results(df):
    # df = pd.read_csv("Results/Results2.csv")
    results = df.iloc[:, 9:]
    cols = results.columns.tolist()
    for col in cols:
        col_vals = np.array(results[col].values.tolist())
        col_vals = [float(x) for x in col_vals]
        results[col] = np.exp(col_vals)
        results[col] = np.divide(results[col], 20)

    final_results = pd.concat([df.iloc[:, 0:9], results], axis=1)
    final_results.to_csv("Results/Results3_Unscaled.csv", index=False)
    return final_results

def scorer(df):
    cols = df.columns[9:].tolist()
    yt = df["MoleFraction"].tolist()

    r2_dict = {}; mse_dict = {}
    for col in cols:
        y_pred = df[col].values.tolist()
        y_pred = [float(x) for x in y_pred]
        r2_dict[col] = r2_score(yt, y_pred)
        mse_dict[col] = mean_squared_error(yt, y_pred)

    score_df = pd.DataFrame([r2_dict, mse_dict]).T
    score_df.columns = ["r2", "mse"]
    score_df.to_csv(f"Results/Stats/AllResults_Scored.csv")

df = pd.read_csv("Results/AllResults.csv")
scorer(df)

def histogram_plotter(dataset, num_features):
    for feature_window in range(0, len(dataset.columns), num_features):
        features_to_plot = dataset.iloc[:, feature_window:(feature_window)+num_features]
        n_rows = 5
        n_cols = 4

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 2*n_rows)) 
        axes = axes.flatten()

        for idx, feature in enumerate(features_to_plot.columns):
            if idx >= len(axes):  
                break
            ax = axes[idx]
            ax.hist(features_to_plot[feature].dropna(), bins=20)
            ax.set_title(f'{feature}')

        plt.tight_layout()
        plt.savefig(f"Results/AllResultsY2.png")
        plt.show()
        plt.clf()

df = pd.read_csv("Results/AllResultsY.csv")
# scorer(df)
df1 = df.iloc[:, 48:]
histogram_plotter(df1, df1.shape[1])


def nested_lists(col):
    nested = []
    for i in range(len(col)):
        nested.append(list(col[i]))
    return nested

def table_to_dict(table): #Converts the tables to dictionaries
    dict = {}
    cols = table.columns.tolist()

    for i in range(len(cols)):
        data = table[cols[i]].values.tolist()
        dict.update({cols[i]: data})

    return dict

# Need to manually aggregate cols because it's not working :((
def aggregator():
    df = pd.read_csv("Results/Results.csv")
    agg_df = df.groupby(["SMILES1", "SMILES2"]).agg(MoleFraction=("MoleFraction", "unique"))
    agg_df.reset_index(inplace=True)

    smiles_groups = df.groupby(["SMILES1", "SMILES2"])
    group_index_dict = smiles_groups.groups #returns dict of {smile pair : indexes}
    keys = list(group_index_dict.keys())

    dict_of_dicts = {}
    for key in keys:
        table = smiles_groups.get_group(key)
        table_dict = table_to_dict(table)
        dict_of_dicts[key] = table_dict
        new_df = pd.DataFrame(table_dict)

    print(len(dict_of_dicts))
    print(keys[0:5])

def pair_scorer(agg_df):
    nested_mf = nested_lists(agg_df["MoleFraction"])
    cols = agg_df.columns[4:].tolist()
    scored_df = agg_df.iloc[:, 0:4]

    for col in cols: 
        r2 = []; mse = []
        nested_y_pred = nested_lists(agg_df[col])
        for i in range(len(nested_mf)):
            try:
                r2.append(r2_score(nested_mf[i], nested_y_pred[i]))
                mse.append(mean_squared_error(nested_mf[i], nested_y_pred[i]))
            except:
                r2.append("NaN")
                mse.append("NaN")
        scored_df.insert(scored_df.shape[1], col, agg_df[col])
        scored_df.insert(scored_df.shape[1], f"{col}-r2", r2)
        scored_df.insert(scored_df.shape[1], f"{col}-mse", mse)

    print(scored_df.shape)
    scored_df.dropna(inplace=True)
    scored_df.reset_index(inplace=True)
    print(scored_df)

    scored_df.to_csv("Results/scored_df.csv", index=False)
