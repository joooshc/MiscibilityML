import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
import random as rand
from tqdm import tqdm
import numpy as np
from joshfunctions import *

def model(pcaX, y):
    params = {'n_estimators': 550, 'max_depth': 5, 'min_samples_split': 7, 'learning_rate': 0.0775, 'loss': "squared_error"}
    mseList = []
    scoreList = []

    model = GradientBoostingRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state = rand.randint(0, 1000))

    for train_index, test_index in tqdm(kf.split(pcaX)):
        X_train, X_test = pcaX[train_index], pcaX[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mseList.append(mean_squared_error(y_test, y_pred))
        scoreList.append(r2_score(y_test, y_pred))
    return mseList, scoreList

def main():
    train1 = pd.read_csv(f"Datasets/TrainTestData/Log/MTrainLDm1.csv")
    train2 = pd.read_csv(f"Datasets/TrainTestData/Log/MTrainLDm2.csv")
    train3 = pd.read_csv(f"Datasets/TrainTestData/Log/MTrainLDm3.csv")

    dataset = pd.concat([train1, train2, train3])
    # dataset = pd.read_csv("Datasets/TrainTestData/WithoutDragon/MTrainQDm.csv")
    components = np.linspace(1, 489, 10, dtype=int)
    filename = filenamegen("GradientBoost", "MTrainLDm")

    y = dataset["MoleFraction"].values #mole fraction
    X = dataset.iloc[0:, 7:].values
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
    eval_df.to_csv(f'Results/GradientBoost/{filename}.csv')

    iteration_plotter(components, eval_df, f"Results/GradientBoost/{filename}")

main()