import numpy as np
import pandas as pd
import warnings, re
from sklearn.decomposition import PCA
from sklearn.svm import SVR
warnings.filterwarnings('ignore')
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from joshfunctions import *

def model(pcaX_train, y_train, pcaX_test, y_test, filename, comp, dataset, save):
    mseList = []; scoreList = []

    SVR_model = SVR(kernel = "rbf", gamma = "auto", C = 2, epsilon = 0.775)

    SVR_model.fit(pcaX_train, y_train)
    y_pred = SVR_model.predict(pcaX_test)

    mseList.append(mean_squared_error(y_test, y_pred))
    scoreList.append(r2_score(y_test, y_pred))

    if save == True:
        scatter_plotter(y_test, y_pred, f'Results/SVR/{filename}', f"Support Vector Regression Model on Unseen Data \n R2: {scoreList[0]:.2f}, RMSE: {np.sqrt(mseList[0]):.2f} \n {dataset} Scaled, PCA {comp}")

        # with open(f'Results/SVR/{folder}-{dataset}-SVR.txt', 'w') as f:
        #     for i in range(len(y_pred)):
        #         f.write(f"{y_pred[i]}, ")

    return mseList, scoreList

def main(folder, dataset):
    print("-"*50)
    print(f"SVR {dataset} {folder}")
    if re.search("NoDragon", folder):
        train = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train.csv")
        test = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Test.csv")
        components = np.linspace(1, 86, 44, dtype=int)
    else:
        train1 = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train1.csv")
        train2 = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train2.csv")
        train3 = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train3.csv")

        train = pd.concat([train1, train2, train3])
        test = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Test.csv")
        components = np.linspace(1, 489, 20, dtype=int)
    
    X_train = train.iloc[:, 7:].values
    y_train = train["MoleFraction"].values
    X_test = test.iloc[:, 7:].values
    y_test = test["MoleFraction"].values

    eval_dict = {}

    filename = filenamegen("SVR", f"{folder}{dataset}TrainTest")

    for i in tqdm(range(len(components))):
        pca = PCA(n_components = components[i])
        pcaX_train = pca.fit_transform(X_train)
        pcaX_test = pca.transform(X_test)

        mseList, scoreList = model(pcaX_train, y_train, pcaX_test, y_test, filename, components[i], dataset, False)
        eval_dict_vals = evaluation(mseList, scoreList)
        eval_dict[components[i]] = eval_dict_vals

    eval_df = pd.DataFrame.from_dict(eval_dict, orient = "index", columns = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"])

    eval_df.to_csv(f"Results/SVR/{filename}.csv")
    print(eval_df)
    # iteration_plotter(components, eval_df, f"Results/SVR/{filename}")

    # eval_df.sort_values(by=["avScore"], ascending=False, inplace=True)
    # comp = eval_df.index[0]
    # pca = PCA(n_components = comp)
    # pcaX_train = pca.fit_transform(X_train)
    # pcaX_test = pca.transform(X_test)
    # mseList, scoreList = model(pcaX_train, y_train, pcaX_test, y_test, filename, comp, dataset, True)

folders = ["NoDragonYx"]
datasets = ["MinMax"]
for folder in folders:
    for dataset in datasets:
        main(folder, dataset)