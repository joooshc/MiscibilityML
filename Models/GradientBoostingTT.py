import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import warnings, re
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import numpy as np
from joshfunctions import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def models(X_train, y_train, X_test, y_test, filename):
    mseList = []; scoreList = []

    params = {'n_estimators': 550, 'max_depth': 5, 'min_samples_split': 7, 'learning_rate': 0.0775, 'loss': "squared_error"}
    model = GradientBoostingRegressor(**params, verbose=0).fit(X_train, y_train)
    y_pred = np.array(model.predict(X_test))

    if not re.search("DragonY", filename):
        if np.max(y_pred) > 1:
            scaler = MinMaxScaler(feature_range=(np.min(y_pred), 1))
            y_pred = scaler.fit_transform(np.array(y_pred).reshape(-1, 1))

    mseList.append(mean_squared_error(y_test, y_pred))
    scoreList.append(r2_score(y_test, y_pred))

    # scatter_plotter(y_test, y_pred, f'Results/GradientBoost/TestSet/{filename}', f"Gradient Boosting Model on Unseen Data (With Dragon)\n R2: {scoreList[0]:.2f}, RMSE: {np.sqrt(mseList[0]):.2f} \n Quantile Scaled, PCA 100")

    # with open(f'Results/GradientBoost/TestSet/MTTQD_Dragon.txt', 'w') as f:
    #     for i in range(len(y_pred)):
    #         f.write(f"{y_pred[i]}, ")

    return mseList, scoreList

def main(folder, dataset):
    print(f"GradientBoost {dataset} {folder}")
    if re.search("NoDragon", folder):
        train = pd.read_csv(f"Datasets/TrainTestData/NoOutlierRemoval/{folder}/{dataset}Train.csv")
        test = pd.read_csv(f"Datasets/TrainTestData/NoOutlierRemoval/{folder}/{dataset}Test.csv")
        components = np.linspace(1, 86, 10, dtype=int)
    else:
        train1 = pd.read_csv(f"Datasets/TrainTestData/NoOutlierRemoval/{folder}/{dataset}Train1.csv")
        train2 = pd.read_csv(f"Datasets/TrainTestData/NoOutlierRemoval/{folder}/{dataset}Train2.csv")
        train3 = pd.read_csv(f"Datasets/TrainTestData/NoOutlierRemoval/{folder}/{dataset}Train3.csv")

        train = pd.concat([train1, train2, train3])
        test = pd.read_csv(f"Datasets/TrainTestData/NoOutlierRemoval/{folder}/{dataset}Test.csv")
        components = np.linspace(1, 489, 20, dtype=int)
    
    y_train = train["MoleFraction"].values
    X_train = train.iloc[0:, 7:].values
    y_test = test["MoleFraction"].values
    X_test = test.iloc[0:, 7:].values
    filename = filenamegen("GradientBoost", f"{folder}{dataset}TrainTest")
    eval_dict = {}

    for i in tqdm(range(len(components))):
        pca = PCA(n_components = components[i])
        pcaX_train = pca.fit_transform(X_train)
        pcaX_test = pca.transform(X_test)
        mseList, scoreList = models(pcaX_train, y_train, pcaX_test, y_test, filename)
        eval_dict_vals = evaluation(mseList, scoreList)
        eval_dict[components[i]] = eval_dict_vals

    eval_cols = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"]
    eval_df = pd.DataFrame(eval_dict).T
    eval_df.columns = eval_cols
    print(eval_df)
    eval_df.to_csv(f'Results/GradientBoost/{filename}.csv')

folders = ["NoDragon", "Dragon"]
datasets = ["Log", "Quantile", "MinMax", "QuantileMinMax"]
for folder in folders:
    for dataset in datasets:
        main(folder, dataset)