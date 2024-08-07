import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def iteration_plotter(components, df, filename):
    plt.clf()
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex = True, figsize = (10, 5))

    ax1.plot(components, df["avScore"], label = "Mean R2 Score")
    ax1.set_ylabel("R2 Score")
    ax1.fill_between(components, df["maxScore"], df["minScore"], alpha = 0.2)
    ax1.legend()
    ax1.set_title("R2")

    ax2.plot(components, df["avMSE"], label = "MSE")
    plt.xlabel("Number of Components")
    ax2.set_title("MSE")
    ax2.fill_between(components, df["maxMSE"], df["minMSE"], alpha = 0.2)
    ax2.legend()

    plt.savefig(f"{filename}.png")
    # plt.show()

def evaluation(mseList, scoreList): #Evaluation metrics
    avScore = np.mean(scoreList)
    scoreCount = sum(i < 0 for i in scoreList)
    mseCount = sum(i > 1 for i in mseList)
    avmse = np.mean(mseList)
    
    eval_dict_vals = [mseCount, max(mseList), min(mseList), np.std(mseList), avmse, scoreCount,  max(scoreList), min(scoreList), np.std(scoreList), avScore]

    return eval_dict_vals

def filenamegen(model, dataset):
    now = datetime.now()
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H%M%S")
    filename = f"{date}-{time}-{model}-{dataset}"
    return filename

def modelnamegen(model, dataset, pca):
    now = datetime.now()
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H%M%S")
    filename = f"{date}-{time}-{model}-{dataset}-{pca}"
    return filename

def scatter_plotter(y_test, y_pred, filename, title):
    plt.style.use("seaborn-v0_8-colorblind")
    plt.clf()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    plt.scatter(y_test, y_pred, alpha=0.9, s=25)
    # plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), '--k', lw=2)
    t = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test, t[0]*y_test + t[1], lw=4, color = 'midnightblue')
    plt.xlabel('True Values', fontsize = 20)
    plt.ylabel('Predictions', fontsize = 20)
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{filename}.png')

def unscaler(col):
    for i in range(len(col)):
        col[i] = np.exp(float(col[i]))
        if col[i] <= 1e-7:
            col[i] = 0
    return col