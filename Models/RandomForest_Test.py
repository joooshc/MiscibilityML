import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

def train_model(X, y):
    component_range = range(1, (X.shape[1] + 1), 5)

    trained_models = {}
    trained_pcas = {}
    for n_components in tqdm(component_range, desc="Processing PCA components"):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        trained_pcas[n_components] = pca

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_pca, y)

        y_pred = model.predict(X_pca)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        print(f"Number of PCA Components: {n_components}")
        print(f"RMSE with Random Forest: {rmse}")
        print(f"R² score with Random Forest: {r2}")
        print("-" * 50)

        trained_models[n_components] = model

    return component_range, trained_pcas, trained_models

def validate_model(X, y, component_range, trained_pcas, trained_models):
    best_r2 = float('-inf')
    best_rmse = float('inf')
    best_y_pred = None
    r2_list = []
    rmse_list = []
    best_pca = []

    for n_components in tqdm(component_range, desc="Validating PCA components"):
        pca = trained_pcas[n_components]
        model = trained_models[n_components]
        X_pca = pca.transform(X)

        y_pred = model.predict(X_pca)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        r2_list.append(r2)
        rmse_list.append(rmse)

        if r2 > best_r2 and rmse < best_rmse:
            best_r2 = r2
            best_rmse = rmse
            best_y_pred = y_pred
            best_pca = n_components
            y_test_ = y
        elif r2 > best_r2:
            best_r2 = r2
            best_y_pred = y_pred
            best_pca = n_components
            y_test_ = y

    print(f"Number of PCA Components: {n_components}")
    print(f"Validation RMSE with Random Forest: {rmse}")
    print(f"Validation R² score with Random Forest: {r2}")
    print("-" * 50)

    return component_range, r2_list, rmse_list, best_y_pred, y_test_, best_pca

def plot_pca(components, r2_list, rmse_list, plot_file_path):
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex = True, figsize = (10, 5))
    ax1.plot(components, r2_list, label = "Mean R2 Score")
    ax1.set_ylabel("R2 Score")
    ax1.legend()
    ax1.set_title("R2")

    ax2.plot(components, rmse_list, label = "MSE")
    plt.xlabel("Number of Components")
    ax2.set_title("MSE")
    ax2.legend()

    plt.savefig(plot_file_path)
    # plt.show() 

def plot_scatter(y_test_, best_y_pred, scatter_file_path, file_name, best_pca):
    slope, intercept = np.polyfit(y_test_, best_y_pred.flatten(), 1)

    r2 = r2_score(y_test_, best_y_pred)
    mse = mean_squared_error(y_test_, best_y_pred)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_, best_y_pred, label="Predictions", alpha=0.5, s=5)
    
    y_fit = slope * y_test_ + intercept
    plt.plot(y_test_, y_fit, '--k', lw=2)

    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Random Forest Model on Unseen Data (Without Dragon)\nR2: {r2:.2f}, RMSE: {np.sqrt(mse):.2f}\n{file_name} Scaled, \nPCA {best_pca}')
    plt.legend()
    plt.savefig(scatter_file_path)
    # plt.show()
    
if __name__ == "__main__":
    
    file_name = ['Log', 'Quantile', 'MinMax', 'QuantileMinMax', 'LogMinMax']
    test_file_name = ['Test1', 'Test2']
    folder_name = ['NoDragon', 'NoDragonY', 'NoDragonYx']
    folder_prefix = ['', 'Y', 'Yx']

    # logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(message)s')
    # print = logging.info 

    for folder, prefix in zip(folder_name, folder_prefix):
        for test_file in test_file_name:
            for i in range(len(file_name)):
                dataset_pt = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Extra/{folder}-stacking/{file_name[i]}Train-stack.csv")
                y_train = dataset_pt['MoleFraction']
                X_train = dataset_pt.iloc[:, 7:-3]

                test_dataset = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Extra/{folder}-stacking/{file_name[i]}{test_file}-stack.csv")
                X_test = test_dataset.iloc[:, 7:-3]
                y_test = test_dataset['MoleFraction']

                # Training
                component_range, trained_pcas, trained_models = train_model(X_train, y_train)

                # Validation
                component_range, r2_list, rmse_list, best_y_pred, y_test_, best_pca = validate_model(X_test, y_test, component_range, trained_pcas, trained_models)        
                # Saving the file
                text_file_path = f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/RF/RandomForest-Stacking/RF_Results{prefix}_{test_file}/{file_name[i]}Test.txt"
                plot_file_path = f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/RF/RandomForest-Stacking/RF_Results{prefix}_{test_file}/{file_name[i]}Test.png"
                scatter_file_path = f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/RF/RandomForest-Stacking/RF_Results{prefix}_{test_file}/sc{file_name[i]}Test.png"
                np.savetxt(text_file_path, best_y_pred.reshape(1, -1), delimiter=",", fmt="s%s")

                # Plotting
                plot_pca(component_range, r2_list, rmse_list, plot_file_path)
                plot_scatter(y_test_, best_y_pred, scatter_file_path, file_name[i], best_pca)