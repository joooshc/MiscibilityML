import catboost as cb
import numpy as np
import pandas as pd
import warnings, re
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from joshfunctions import *

def models(X_train, X_test, y_train, y_test, filename, dataset, folder, comp, save):
    mseList = []; scoreList = []
    train = cb.Pool(X_train, y_train)
    test = cb.Pool(X_test, y_test)

    model = cb.CatBoostRegressor(iterations = 200,
                        learning_rate = 0.05,
                        depth = 7,
                        l2_leaf_reg = 0.2,
                        loss_function = 'RMSE',
                        logging_level='Silent')
    model.fit(train)
    y_pred = model.predict(X_test)

    mseList.append(mean_squared_error(y_test, y_pred))
    scoreList.append(r2_score(y_test, y_pred))
    if save == True:
        # scatter_plotter(y_test, y_pred, f'Results/CatBoost/{filename}', f"CatBoost Model on Unseen Data\n R2: {scoreList[0]:.2f}, RMSE: {np.sqrt(mseList[0]):.2f} \n {dataset} Scaled, PCA {comp}")

        with open(f'Results/CatBoost/Stacking/{folder}/Test2/{dataset}-CatBoost.txt', 'w') as f:
            for i in range(len(y_pred)):
                f.write(f"{y_pred[i]}, ")
    return mseList, scoreList

def main(folder, dataset):
    print(f"CatBoost {dataset} {folder}")
    if re.search("NoDragon", folder):
        train = pd.read_csv(f"Datasets/TrainTestData/Extra/{folder}-stacking/{dataset}Train-stack.csv")
        test = pd.read_csv(f"Datasets/TrainTestData/Extra/{folder}-stacking/{dataset}Test2-stack.csv")
        components = np.linspace(1, 83, 20, dtype=int)
    else:
        train1 = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train1.csv")
        train2 = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train2.csv")
        train3 = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Train3.csv")

        train = pd.concat([train1, train2, train3])
        test = pd.read_csv(f"Datasets/TrainTestData/{folder}/{dataset}Test.csv")
        components = np.linspace(1, 489, 20, dtype=int)

    filename = filenamegen("CatBoost", f"{folder}{dataset}TrainTest")

    y_train = train["MoleFraction"].values
    X_train = train.iloc[0:, 7:].values
    y_test = test["MoleFraction"].values
    X_test = test.iloc[0:, 7:].values

    eval_dict = {}

    for i in tqdm(range(len(components))):

        pca = PCA(n_components = components[i])
        pcaX_train = pca.fit_transform(X_train)
        pcaX_test = pca.transform(X_test)
        mseList, scoreList = models(pcaX_train, pcaX_test, y_train, y_test, filename, dataset, folder, components[i], False)
        eval_dict_vals = evaluation(mseList, scoreList)
        eval_dict[components[i]] = eval_dict_vals

    eval_cols = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"]
    eval_df = pd.DataFrame(eval_dict).T
    eval_df.columns = eval_cols
    print(eval_df)

    eval_df.to_csv(f'Results/CatBoost/Stacking/{folder}/Test2/{filename}.csv')
    iteration_plotter(components, eval_df, f'Results/CatBoost/Stacking/{folder}/Test2/{filename}')

    eval_df.sort_values(by=["avScore"], ascending=False, inplace=True)

    comp = eval_df.index[0]
    pca = PCA(n_components = comp)
    pcaX_train = pca.fit_transform(X_train)
    pcaX_test = pca.transform(X_test)
    mseList, scoreList = models(pcaX_train, pcaX_test, y_train, y_test, filename, dataset, folder, comp, True)

folders = ["NoDragon", "NoDragonYx"]
datasets = ["Log", "Quantile", "MinMax", "QuantileMinMax", "LogMinMax"]
for folder in folders:
    for dataset in datasets:
        main(folder, dataset)