import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from joshfunctions import *
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import random as rand
from sklearn.decomposition import PCA

def model(pcaX, y):
    mseList = []; scoreList = []

    model = lgb.LGBMRegressor(objective = "regression", verbose=0, n_jobs=-1)
    
    kf = KFold(n_splits = 5, shuffle = True, random_state = rand.randint(0, 1000))

    for train_index, test_index in kf.split(pcaX):
        X_train, X_test = pcaX[train_index], pcaX[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mseList.append(mean_squared_error(y_test, y_pred))
        scoreList.append(r2_score(y_test, y_pred))
    return mseList, scoreList

def main():
    dataset = pd.read_csv("Datasets/TrainTestData/NoDragon/LogTrain.csv")
    components = np.linspace(1, 86, 20, dtype=int)

    filename = filenamegen("LightGBM", "NoDragonLogTrain1-86")
    y = dataset["MoleFraction"].values #mole fraction
    X = dataset.iloc[:, 7:].values
    eval_dict = {}

    for i in tqdm(range(len(components))):
        pca = PCA(n_components = components[i])
        pcaX = pca.fit_transform(X)
        mseList, scoreList = model(pcaX, y)
        eval_dict_vals = evaluation(mseList, scoreList)
        eval_dict[components[i]] = eval_dict_vals

    eval_cols = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"]
    eval_df = pd.DataFrame(eval_dict).T
    eval_df.columns = eval_cols

    print(eval_df)

    eval_df.to_csv(f'Results/LightGBM/{filename}.csv')
    iteration_plotter(components, eval_df, f'Results/LightGBM/{filename}')

main()