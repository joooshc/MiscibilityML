import pandas as pd
import lightgbm as lgb
import warnings, re
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import numpy as np
from joshfunctions import *
import numpy as np

def models(X_train, y_train, X_test, y_test, filename, booster, folder, dataset, comp, save):
    mseList = []; scoreList = []

    model = lgb.LGBMRegressor(**booster)
    model.fit(X_train, y_train)
    y_pred = np.array(model.predict(X_test, verbose=0))

    mseList.append(mean_squared_error(y_test, y_pred))
    scoreList.append(r2_score(y_test, y_pred))

    if save == True:
        # scatter_plotter(y_test, y_pred, f'Results/LightGBM/{filename}', f"LightGBM {booster['boosting_type']}\n R2: {scoreList[0]:.2f}, RMSE: {np.sqrt(mseList[0]):.2f} \n {dataset} Scaled, PCA {comp}")
        # scatter_plotter(y_test, y_pred, f'Results/PlotsForReport/{filename}', f"LightGBM Gradient Boosted Decision Trees\n R²: {scoreList[0]:.2f}, MSE: {mseList[0]:.2f} \n MinMax Scaled Features & Log Scaled Targets, PCA {comp}")

        with open(f'Results/LightGBM/Immiscible/{dataset}-{booster["boosting_type"]}.txt', 'w') as f:
            for i in range(len(y_pred)):
                f.write(f"{y_pred[i]}, ")

    return mseList, scoreList

def main(folder, dataset, booster):
    print(f"LightGBM {booster['boosting_type']} {dataset} {folder}")

    train = pd.read_csv(f"Datasets/TrainTestData/ImmiscibleSets/{dataset}Train.csv")
    test = pd.read_csv(f"Datasets/TrainTestData/ImmiscibleSets/{dataset}Test.csv")
    components = np.linspace(1, 86, 20, dtype=int)
    
    y_train = train["MoleFraction"].values
    X_train = train.iloc[0:, 7:].values
    y_test = test["MoleFraction"].values
    X_test = test.iloc[0:, 7:].values
    filename = filenamegen("LightGBM", f"{booster['boosting_type']}-{dataset}TrainTest")
    eval_dict = {}

    for i in tqdm(range(len(components))):
        pca = PCA(n_components = components[i])
        pcaX_train = pca.fit_transform(X_train)
        pcaX_test = pca.transform(X_test)
        mseList, scoreList = models(pcaX_train, y_train, pcaX_test, y_test, filename, booster, folder, dataset, components[i], False)
        eval_dict_vals = evaluation(mseList, scoreList)
        eval_dict[components[i]] = eval_dict_vals

    eval_cols = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"]
    eval_df = pd.DataFrame(eval_dict).T
    eval_df.columns = eval_cols
    print(eval_df)
    eval_df.to_csv(f'Results/LightGBM/Immiscible/{filename}.csv')
    iteration_plotter(components, eval_df, f'Results/LightGBM/Immiscible/{filename}')

    eval_df.sort_values(by=["avScore"], ascending=False, inplace=True)

    comp = eval_df.index[0]
    pca = PCA(n_components = comp)
    pcaX_train = pca.fit_transform(X_train)
    pcaX_test = pca.transform(X_test)
    mseList, scoreList = models(pcaX_train, y_train, pcaX_test, y_test, filename, booster, folder, dataset, comp, True)

dart = {
    "boosting_type": "dart",
    "learning_rate": 0.05,
    "n_estimators" : 1000,
    "objective": "regression",
    "verbose": 0,
    "n_jobs": -1,
    "force_col_wise": True,
    "reg_alpha": 0.5,
    "reg_lambda": 0.2,
}
gbdt = {
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "n_estimators" : 600,
    "objective": "regression",
    "verbose": 0,
    "n_jobs": -1,
    "force_col_wise": True,
    "reg_alpha": 1,
    "reg_lambda": 0,
}
goss = {
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "n_estimators" : 600,
    "objective": "regression",
    "verbose": 0,
    "n_jobs": -1,
    "force_col_wise": True,
    "reg_alpha": 1,
    "reg_lambda": 0.1,
    "data_sample_strategy" : "goss"
}
rf = {
    "boosting_type": "rf",
    "learning_rate": 0.01,
    "bagging_freq" : [1],
    "bagging_fraction" :[0.7],
    "n_estimators" : 500,
    "objective": "regression",
    "verbose": 0,
    "n_jobs": -1,
    "force_col_wise": True,
    "reg_alpha": 1,
    "reg_lambda": 1
}

folders = ["Immiscible"]
datasets = ["Log", "Quantile", "MinMax", "LogMinMax", "QuantileMinMax"]
boosters = [gbdt, dart, rf]
for booster in boosters:
    for folder in folders:
        for dataset in datasets:
            main(folder, dataset, booster)