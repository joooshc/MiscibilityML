import catboost as cb
import numpy as np
import pandas as pd
import warnings, time
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
import random as rand
from tqdm import tqdm
from sklearn.model_selection import KFold
from joshfunctions import *

def model(pcaX, y):
    mseList = []; scoreList = []

    model = cb.CatBoostRegressor(iterations = 500,
                    learning_rate = 0.05,
                    depth = 7,
                    l2_leaf_reg = 0.2,
                    loss_function = 'RMSE',
                    logging_level='Silent')
    
    kf = KFold(n_splits = 5, shuffle = True, random_state = rand.randint(0, 1000))

    for train_index, test_index in kf.split(pcaX):
        X_train, X_test = pcaX[train_index], pcaX[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train = cb.Pool(X_train, y_train)

        model.fit(train)

        y_pred = model.predict(X_test)
        mseList.append(mean_squared_error(y_test, y_pred))
        scoreList.append(r2_score(y_test, y_pred))
    return mseList, scoreList

def main():
    start_time = time.time()
    train1 = pd.read_csv("Datasets/TrainTestData/Dragon/LogTrain1.csv")
    train2 = pd.read_csv("Datasets/TrainTestData/Dragon/LogTrain2.csv")
    train3 = pd.read_csv("Datasets/TrainTestData/Dragon/LogTrain3.csv")

    dataset = pd.concat([train1, train2, train3])
    # dataset = pd.read_csv("Datasets/TrainTestData/NoDragon/LogTrain.csv")
    components = np.linspace(1, 86, 20, dtype=int)

    filename = filenamegen("CatBoost", "DLogTrain89")
    y = dataset["MoleFraction"].values
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

    eval_df.to_csv(f'Results/CatBoost/{filename}.csv')
    iteration_plotter(components, eval_df, f'Results/CatBoost/{filename}')
    elapsed_time = time.time() - start_time
    with open(f'Results/CatBoost/{filename}.txt', 'w') as f:
        f.write(f"Time taken: {elapsed_time} seconds")

main()