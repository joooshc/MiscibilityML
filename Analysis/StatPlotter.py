import pandas as pd
import matplotlib.pyplot as plt
import re, time
import numpy as np

def index_finder(df, to_find):
    models = df["Model"].values.tolist()
    index_dict = {}
    for find in to_find:
        indexes = []
        for i in range(0, len(models)):
            # if re.search(f"-{find}$", models[i]):
            if re.search(find, models[i]):
                indexes.append(i)
        index_dict[find] = [np.min(indexes), np.max(indexes)+1]
    return index_dict

def filter_by_scaler(df, dataset, index_dict):
    output = pd.DataFrame()
    x = 0
    for i in range(8):
        indexes = [x+index_dict[dataset][0], x+index_dict[dataset][1]]
        selection = df.iloc[indexes[0]:indexes[1]]
        output = pd.concat([output, selection])
        x = x + len(df)//8
    output.reset_index(drop=True, inplace=True)
    print(output.shape, dataset)
    output.to_csv(f"Results/{dataset}-stats.csv", index=False)
    return output

def filter_by_algorithm(df, dataset, index_dict):
    indexes = [index_dict[dataset][0], index_dict[dataset][1]]
    output = df.iloc[indexes[0]:indexes[1]]
    output.reset_index(drop=True, inplace=True)
    print(output.shape, dataset)
    output.to_csv(f"Results/{dataset}Y-stats.csv", index=False)
    return output

def stat_fetcher(dataset, stat):
    df = pd.read_csv(f"Results/Stats/{dataset}-stats.csv")
    return df[stat].values.tolist()

def plot_both_hist(stat):
    fig, ax = plt.subplots(figsize=(16, 9))
    r2_y = stat_fetcher("AllResultsY", stat)
    r2_x = stat_fetcher("AllResults", stat)
    r2_y = [x for x in r2_y if x > -5]
    r2_x = [x for x in r2_x if x > -5]
    ax.hist(r2_y, bins=30, alpha=1, label="Log Scaled Mole Fractions")
    ax.hist(r2_x, bins=30, alpha=0.8, label="Unscaled Mole Fractions")

    plt.legend(fontsize=16)
    plt.title("R² Distribution", fontsize=24)
    plt.xlabel("R²", fontsize=16)
    plt.ylabel("Frequency Density", fontsize=20)
    plt.yticks(fontsize=14); plt.xticks(fontsize=14)
    plt.tight_layout()

def hist_by_algorithm():
    fig, ax = plt.subplots(figsize=(16, 9))
    for algorithm in algorithms:
        try:
            data = stat_fetcher(algorithm, "R2")
            data = [x for x in data if x > -5]
            ax.hist(data, bins=30, alpha=0.8, label=algorithm)
        except:
            pass

    plt.legend(fontsize=16)
    plt.title("R² Distribution for Unscaled Mole Fractions", fontsize=24)
    plt.xlabel("R²", fontsize=16)
    plt.ylabel("Frequency Density", fontsize=20)
    plt.yticks(fontsize=14); plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("Results/PlotsForReport/R²DistByAlgorithm.png", dpi=300)
    plt.show()

def percentages_df(x, categories, metric):
    x = np.array(x)
    categories = [round(x, 2) for x in categories]
    percentages_dict = {}
    x_total = len(x)
    # percentages_dict[f"{metric}<0"] = (len([y for y in x if y < 0])/x_total)*100
    for cat in categories:
        subsect_x = [y for y in x if y > cat]
        percentages_dict[f">{cat}"] = ((len(subsect_x)/x_total)*100) 

    df = pd.DataFrame.from_dict(percentages_dict, orient="index")
    return df

if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-colorblind")

    stats_y = pd.read_csv("Results/Stats/AllResultsY-stats.csv")
    stats_x = pd.read_csv("Results/Stats/AllResults-stats.csv")

    datasets = ["Log", "MinMax", "Quantile", "QuantileMinMax", "LogMinMax"]
    datasetsY = [x + "Y" for x in datasets]
    algorithms = ["dart", "gbdt", "rf", "goss", "CatBoost", "SVR", "DNeuralNetwork"]
    algorithmsY = [x + "Y" for x in algorithms]

    # index_dict = index_finder(stats_y, algorithms)
    # for algorithm in algorithms:
    #     filter_by_algorithm(stats_y, algorithm, index_dict)

    r2_y = stat_fetcher("AllResultsY", "R_squared")
    r2_x = stat_fetcher("AllResults", "R_squared")
    mse_y = stat_fetcher("AllResultsY", "MSE")
    mse_x = stat_fetcher("AllResults", "MSE")

    cat = [0, 0.01]
    categories2 = (np.linspace(0.2, 1, 5))
    categories = cat + list(categories2)

    r2y_df = percentages_df(r2_y, categories, "R2")
    r2x_df = percentages_df(r2_x, categories, "R2")
    msey_df = percentages_df(mse_y, categories, "R2")
    msex_df = percentages_df(mse_x, categories, "R2")
    all_r2 = pd.concat([r2y_df, r2x_df, msey_df, msex_df], axis=1)
    all_r2.columns = ["r2 log", "r2", "mse log", "mse"]
    print(all_r2)
    # all_r2.to_csv("Results/Stats/MSEPercentages.csv")

# R²