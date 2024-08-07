import pandas as pd

def counter(models):
    scaler_dict = {}
    algorithm_dict = {}

    for scaler in scalers:
        s = 0
        for model in models:
            model = model.split("-")
            m_scaler = model[1]
            if m_scaler == scaler:
                s += 1
        scaler_dict[scaler] = s

    for algorithm in algorithms:
        a = 0
        for model in models:
            model = model.split("-")
            m_algorithm = model[0]
            if m_algorithm == algorithm:
                a += 1
        algorithm_dict[algorithm] = a

    scaler_df = pd.DataFrame.from_dict(scaler_dict, orient="index", columns=["Count"])
    algorithm_df = pd.DataFrame.from_dict(algorithm_dict, orient="index", columns=["Count"])

    return scaler_df, algorithm_df

def totaler(df):
    col1 = df["CountY"].values.tolist()
    col2 = df["CountX"].values.tolist()
    total = [sum(x) for x in zip(col1, col2)]
    df.insert(2, "Total", total)
    df.sort_values(by=["Total"], ascending=False, inplace=True)
    return df

if __name__ == "__main__":
    dfY = pd.read_csv("Results/Stats/AllResultsY_Unique-stats.csv")
    dfX = pd.read_csv("Results/Stats/AllResults_Unique-stats.csv")

    modelsY = dfY["Model"].tolist()
    modelsX = dfX["Model"].tolist()

    scalers = ["MinMax", "Quantile", "Log", "QuantileMinMax", "LogMinMax"]
    algorithms = ["goss", "gbdt", "dart", "rf", "CatBoost", "SVR", "DNeuralNetwork", "RandomForest"]

    sdfY, adfY = counter(modelsY)
    sdfX, adfX = counter(modelsX)

    scaler_df = pd.concat([sdfY, sdfX], axis=1)
    algorithm_df = pd.concat([adfY, adfX], axis=1)
    scaler_df.columns = ["CountY", "CountX"]; algorithm_df.columns = ["CountY", "CountX"]

    scaler_df = totaler(scaler_df)
    algorithm_df = totaler(algorithm_df)

    print(scaler_df)
    print(algorithm_df)

    scaler_df.to_csv("Results/Stats/ScalerCount.csv")
    algorithm_df.to_csv("Results/Stats/AlgorithmCount.csv")