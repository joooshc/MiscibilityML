import pandas as pd
import lightgbm as lgb
import warnings, re
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import numpy as np
from joshfunctions import *
from sklearn.model_selection import GridSearchCV
import numpy as np

def models(X_train, y_train, X_test, y_test, grid):
    mseList = []; scoreList = []

    model = GridSearchCV(estimator = lgb.LGBMRegressor(), param_grid = grid, verbose=0, n_jobs=-1, scoring="r2")
    model.fit(X_train, y_train)
    y_pred = np.array(model.predict(X_test))
    best_params = model.best_params_
    mseList.append(mean_squared_error(y_test, y_pred))
    scoreList.append(r2_score(y_test, y_pred))
    return mseList, scoreList, best_params, model.cv_results_

def main(folder, dataset):
    print(f"LightGBM {dataset} {folder}")
    if re.search("NoDragon", folder):
        train = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train.csv")
        test = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Test.csv")
        components = np.linspace(1, 86, 10, dtype=int)
    else:
        train1 = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train1.csv")
        train2 = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train2.csv")
        train3 = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train3.csv")

        train = pd.concat([train1, train2, train3])
        test = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Test.csv")
        components = np.linspace(1, 489, 20, dtype=int)
    
    y_train = train["MoleFraction"].values
    X_train = train.iloc[0:, 7:].values
    y_test = test["MoleFraction"].values
    X_test = test.iloc[0:, 7:].values
    filename = filenamegen("LightGBM", f"{folder}{dataset}TrainTestOPT")
    eval_dict = {}
    components = [86]

    grid = {
        "objective" : ["regression"],
        "boosting_type" : ["rf"],
        "bagging_freq" : [1],
        "bagging_fraction" :[0.7],
        # "learning_rate" : [0.01, 0.05, 0.1, 0.2],
        "learning_rate" : [0.01],
        "n_estimators" : np.linspace(100, 1000, 10, dtype=int)
        # "reg_alpha" : np.linspace(0, 1, 11),
        # "reg_lambda" : np.linspace(0, 1, 11)
    }

    for i in tqdm(range(len(components))):
        pca = PCA(n_components = components[i])
        pcaX_train = pca.fit_transform(X_train)
        pcaX_test = pca.transform(X_test)
        mseList, scoreList, best_params, results = models(pcaX_train, y_train, pcaX_test, y_test, grid)
        eval_dict_vals = evaluation(mseList, scoreList)
        eval_dict[components[i]] = eval_dict_vals

        best_params["n_components"] = components[i]
        bpdf = pd.DataFrame(best_params, index=[0])
        bpdf.to_csv(f'Results/LightGBM/rf{filename}BestParams.csv')
        print(bpdf)

        resultsdf = pd.DataFrame(results)
        resultsdf.to_csv(f'Results/LightGBM/rf{filename}Results.csv')

    eval_cols = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"]
    eval_df = pd.DataFrame(eval_dict).T
    eval_df.columns = eval_cols
    print(eval_df)
    eval_df.to_csv(f'Results/LightGBM/rf{filename}.csv')

main("NoDragon", "Quantile")