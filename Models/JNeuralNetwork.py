import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA
from tqdm import tqdm
from joshfunctions import *
from sklearn.metrics import mean_squared_error, r2_score
import random as rand
from concurrent.futures import ProcessPoolExecutor

def model(pcaX, y, comp):
    mseList = []; scoreList = []
    model = Sequential()
    optimiser = tf.keras.optimizers.Adam()
    model.add(Dense(64, input_dim= comp, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimiser, loss="mse", metrics=["mse"])

    kf = KFold(n_splits=5, shuffle=True, random_state = rand.randint(0, 1000))

    for train_index, test_index in kf.split(pcaX):
        X_train, X_test = pcaX[train_index], pcaX[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train, epochs=150, verbose=0, batch_size=32)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)

        mseList.append(mse)
        scoreList.append(score)
    
    return mseList, scoreList

def main():
    train1 = pd.read_csv("Datasets/TrainTestData/LinearMF/Train_LinearMF_QMM1.csv")
    train2 = pd.read_csv("Datasets/TrainTestData/LinearMF/Train_LinearMF_QMM2.csv")
    train3 = pd.read_csv("Datasets/TrainTestData/LinearMF/Train_LinearMF_QMM3.csv")

    dataset = pd.concat([train1, train2, train3])
    components = np.linspace(1, 489, 20, dtype=int)
    # dataset = pd.read_csv("Datasets/TrainTestData/WithoutDragon/MTrainLDm_LinearMF.csv")
    # components = np.linspace(1, 84, 20, dtype=int)
    y = dataset["MoleFraction"].values
    X = dataset.iloc[:, 7:].values
    filename = filenamegen("JNeuralNetwork", "Train_LinearMF_QMM1-489")
    eval_dict = {}

    for i in tqdm(range(len(components))):
        pca = PCA(n_components = components[i])
        pcaX = pca.fit_transform(X)
        mseList, scoreList = model(pcaX, y, components[i])
        eval_dict_vals = evaluation(mseList, scoreList)
        eval_dict[components[i]] = eval_dict_vals
    
    eval_cols = ["mseCount", "maxMSE", "minMSE", "stdMSE", "avMSE", "scoreCount", "maxScore", "minScore", "stdScore", "avScore"]
    eval_df = pd.DataFrame(eval_dict).T
    eval_df.columns = eval_cols

    print(eval_df)
    eval_df.to_csv(f'Results/JNeuralNetwork/{filename}.csv')
    iteration_plotter(components, eval_df, f'Results/JNeuralNetwork/{filename}')

main()