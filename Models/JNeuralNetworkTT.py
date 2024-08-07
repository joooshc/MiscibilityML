import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from joshfunctions import *
from sklearn.metrics import mean_squared_error, r2_score
import re, time
from sklearn.preprocessing import MinMaxScaler

def model(X_train, X_test, y_train, y_test, comp, filename):
    start_time = time.time()
    mseList = []; scoreList = []
    model = Sequential()
    optimiser = tf.keras.optimizers.Adam()
    model.add(Dense(64, input_dim= comp, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=optimiser, loss="mae", metrics=["mse"])

    model.fit(X_train, y_train, epochs=150, verbose=0, batch_size=32)
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred).flatten()
    scaler = MinMaxScaler(feature_range=(0, 1))
    if not re.search("DragonY", filename):
        if np.max(y_test) > 1:
            y_pred = scaler.fit_transform(y_pred.reshape(-1, 1))

    mse = mean_squared_error(y_test, y_pred)
    score = r2_score(y_test, y_pred)
    elapsed_time = time.time() - start_time

    print(f"{filename}\n PCA: {comp} R2: {score:.2f}, MSE: {mse:.2f} \n Elapsed time: {elapsed_time:.2f} seconds")
    print("-"*50)

    mseList.append(mse)
    scoreList.append(score)
    
    # scatter_plotter(y_test, y_pred, f'Results/JNeuralNetwork/{filename}', f"Deep Neural Network Model on Unseen Data (With Dragon)\n R2: {scoreList[0]:.2f}, RMSE: {np.sqrt(mseList[0]):.2f} \n Log Scaled, PCA 150")

    # with open(f'Results/JNeuralNetwork/MTTQD_Dragon2.txt', 'w') as f:
    #     for i in range(len(y_pred)):
    #         f.write(f"{y_pred[i]}, ")

    return mseList, scoreList

def main(folder, dataset):
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
    print("-"*50)
    print(f"Neural Network {dataset} {folder}")
    if re.search("NoDragon", folder):
        train = pd.read_csv(f"Datasets/TrainTestData/Augmented/Gamma/{folder}/{dataset}Train.csv")
        test = pd.read_csv(f"Datasets/TrainTestData/Augmented/Gamma/{folder}/{dataset}Test.csv")
        components = np.linspace(1, 86, 20, dtype=int)
        del test["MinTemp"];del test["MaxTemp"]
    else:
        train1 = pd.read_csv(f"Datasets/TrainTestData/Augmented/Gamma/{folder}/{dataset}Train1.csv")
        train2 = pd.read_csv(f"Datasets/TrainTestData/Augmented/Gamma/{folder}/{dataset}Train2.csv")
        train3 = pd.read_csv(f"Datasets/TrainTestData/Augmented/Gamma/{folder}/{dataset}Train3.csv")
        train4 = pd.read_csv(f"Datasets/TrainTestData/Augmented/Gamma/{folder}/{dataset}Train4.csv")
        train5 = pd.read_csv(f"Datasets/TrainTestData/Augmented/Gamma/{folder}/{dataset}Train5.csv")
        train6 = pd.read_csv(f"Datasets/TrainTestData/Augmented/Gamma/{folder}/{dataset}Train6.csv")

        train = pd.concat([train1, train2, train3, train4, train5, train6])
        test = pd.read_csv(f"Datasets/TrainTestData/Augmented/Gamma/{folder}/{dataset}Test.csv")
        components = np.linspace(1, 489, 20, dtype=int)
        del test["MinTemp"];del test["MaxTemp"]

    y_train = train["MoleFraction"].values
    X_train = train.iloc[0:, 5:].values
    y_test = test["MoleFraction"].values
    X_test = test.iloc[0:, 5:].values

    filename = filenamegen("JNeuralNetwork", f"{folder}{dataset}TrainTest")
    eval_dict = {}
    for i in tqdm(range(len(components))):
        pca = PCA(n_components = components[i])
        pcaX_train = pca.fit_transform(X_train)
        pcaX_test = pca.transform(X_test)
        mseList, scoreList = model(pcaX_train, pcaX_test, y_train, y_test, components[i], filename)
        eval_dict_vals = evaluation(mseList, scoreList)
        eval_dict[components[i]] = eval_dict_vals
    
    eval_cols = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"]
    eval_df = pd.DataFrame(eval_dict).T
    eval_df.columns = eval_cols
    print(eval_df)
    eval_df.to_csv(f'Results/JNeuralNetwork/aug{filename}.csv')
    iteration_plotter(components, eval_df, f'Results/JNeuralNetwork/aug{filename}')

folders = ["NoDragon", "Dragon"]
datasets = ["Log", "Quantile", "MinMax", "QuantileMinMax"]
for folder in folders:
    for dataset in datasets:
        main(folder, dataset)