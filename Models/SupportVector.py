import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
warnings.filterwarnings('ignore')
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import random as rand
from joshfunctions import *

def model(pcaX, y, grid):
    mseList = []; scoreList = []

    SVR_model = SVR(**grid)
    kf = KFold(n_splits = 5, shuffle = True, random_state = rand.randint(0, 1000))

    for train_index, test_index in kf.split(pcaX):
        X_train, X_test = pcaX[train_index], pcaX[test_index]
        y_train, y_test = y[train_index], y[test_index]

        SVR_model.fit(X_train, y_train)
        y_pred = SVR_model.predict(X_test)

        mseList.append(mean_squared_error(y_test, y_pred))
        scoreList.append(r2_score(y_test, y_pred))

    return mseList, scoreList

def main():
    # train1 = pd.read_csv("Datasets/TrainTestData/LinearMF/Train_LinearMF_QMM1.csv")
    # train2 = pd.read_csv("Datasets/TrainTestData/LinearMF/Train_LinearMF_QMM2.csv")
    # train3 = pd.read_csv("Datasets/TrainTestData/LinearMF/Train_LinearMF_QMM3.csv")

    # dataset = pd.concat([train1, train2, train3])

    dataset = pd.read_csv("Datasets/TrainTestData/NoDragonYx/MinMaxTrain.csv")
    components = np.linspace(1, 86, 44, dtype=int)
    y = dataset["MoleFraction"].values
    X = dataset.iloc[:, 7:].values
    eval_dict = {}

    grid = {"kernel": "rbf",
            "gamma": "auto",
            "C": 2.0,
            "epsilon": 0.775}
    filename = filenamegen("SVR", "NoDragonLogTrain1-86")

    for i in tqdm(range(len(components))):
        pca = PCA(n_components = components[i])
        pcaX = pca.fit_transform(X)

        mseList, scoreList = model(pcaX, y, grid)
        eval_dict_vals = evaluation(mseList, scoreList)
        eval_dict[components[i]] = eval_dict_vals

    eval_df = pd.DataFrame.from_dict(eval_dict, orient = "index", columns = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"])
    eval_df.to_csv(f"Results/SVR/{filename}.csv")
    print(eval_df)
    iteration_plotter(components, eval_df, f"Results/SVR/{filename}")

main()