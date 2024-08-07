import numpy as np
import matplotlib.pyplot as plt
import random, torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from tqdm import tqdm

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)

def train_model(X, y, device):
    # Set random seeds
    np.random.seed(random.randint(0, 1000))
    torch.manual_seed(random.randint(0, 1000))
    if torch.cuda.is_available():
        print("\ncuda is available\n")
        torch.cuda.manual_seed_all(random.randint(0, 1000))

    component_range = range(1, (X.shape[1] + 1), 1)

    trained_models = {}
    trained_pcas = {}
    for n_components in tqdm(component_range, desc=f"Processing PCA components"):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        trained_pcas[n_components] = pca

        X_train, X_test = torch.tensor(X_pca).float().to(device), torch.tensor(X_pca).float().to(device)
        y_train, y_test = torch.tensor(y.values).float().to(device), torch.tensor(y.values).float().to(device)

        model = NeuralNetwork(input_dim=n_components).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        trained_models[n_components] = model.state_dict()

        best_loss = float('inf')
        patience = 500
        no_improve_epochs = 0

        for epoch in range(150):
            # Training
            optimizer.zero_grad()
            outputs = model(X_train).squeeze()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Checking if training loss improved
            if loss.item() < best_loss - 1e-5:  # Using a small constant of 1e-5 to determine improvement
                best_loss = loss.item()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Check for early stopping
            if no_improve_epochs >= patience:
                print(f"Stopped early at epoch {epoch}")
                break

        y_pred = model(X_test).cpu().detach().numpy()
        y_test_cpu = y_test.cpu().numpy()

        rmse = np.sqrt(mean_squared_error(y_test_cpu, y_pred))
        r2 = r2_score(y_test_cpu, y_pred)

        print(f"Number of PCA Components: {n_components}")
        print(f"RMSE with Neural Network: {rmse}")
        print(f"R^2 score with Neural Network: {r2}")
        print("-" * 50)

    return component_range, trained_pcas, trained_models

def validate_model(X, y, component_range, trained_pcas, trained_models, device):
    best_r2 = float('-inf')
    best_rmse = float('inf')
    best_y_pred = None
    best_pca = None
    r2_list = []
    rmse_list = []

    for n_components in tqdm(component_range, desc=f"Validating PCA components"):
        pca = trained_pcas[n_components]
        model_state = trained_models[n_components]
        X_pca = pca.transform(X)

        # Convert data to PyTorch tensors and move to device
        X_test = torch.tensor(X_pca).float().to(device)
        y_test = torch.tensor(y.values).float().to(device)

        model = NeuralNetwork(input_dim=n_components).to(device)
        model.load_state_dict(model_state)

        y_pred = model(X_test).cpu().detach().numpy()
        y_test_cpu = y_test.cpu().numpy()

        rmse = np.sqrt(mean_squared_error(y_test_cpu, y_pred))
        r2 = r2_score(y_test_cpu, y_pred)

        r2_list.append(r2)
        rmse_list.append(rmse)

        # Jointly considering R² and RMSE
        if r2 > best_r2 and rmse < best_rmse:
            best_r2 = r2
            best_rmse = rmse
            best_y_pred = y_pred
            best_pca = n_components
            y_test_ = y_test_cpu
        elif r2 > best_r2:
            best_r2 = r2
            best_y_pred = y_pred
            best_pca = n_components
            y_test_ = y_test_cpu

        print(f"Number of PCA Components: {n_components}")
        print(f"Validation RMSE with Neural Network: {rmse}")
        print(f"Validation R^2 score with Neural Network: {r2}")
        print("-" * 50)

    return component_range, r2_list, rmse_list, best_y_pred, y_test_, best_pca

def plot_pca(components, r2_list, rmse_list):
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex = True, figsize = (10, 5))
    ax1.plot(components, r2_list, label = "Mean R2 Score")
    ax1.set_ylabel("R2 Score")
    ax1.legend()
    ax1.set_title("R2")

    ax2.plot(components, rmse_list, label = "MSE")
    plt.xlabel("Number of Components")
    ax2.set_title("MSE")
    ax2.legend()
    plt.show() 

def plot_scatter(y_test_, best_y_pred, best_pca):
    slope, intercept = np.polyfit(y_test_, best_y_pred.flatten(), 1)

    r2 = r2_score(y_test_, best_y_pred)
    mse = mean_squared_error(y_test_, best_y_pred)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_, best_y_pred, label="Predictions", alpha=0.5, s=5)
    
    y_fit = slope * y_test_ + intercept
    plt.plot(y_test_, y_fit, '--k', lw=2)

    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Deep Neural Network Model on Unseen Data (Without Dragon)\nR2: {r2:.2f}, RMSE: {np.sqrt(mse):.2f}\nStacked Results as Features / Unscaled Targets, \nPCA {best_pca}')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_test_path = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/Stacking/"
    train_dataset = pd.read_csv(f"{train_test_path}TrainYx.csv")
    test_dataset = pd.read_csv(f"{train_test_path}TestYx.csv")
   
    y_train = train_dataset['MoleFraction']
    X_train = train_dataset.iloc[:, 8:]

    y_test = test_dataset['MoleFraction']
    X_test = test_dataset.iloc[:, 8:]

    # Training
    component_range, trained_pcas, trained_models = train_model(X_train, y_train, device)

    # Validation
    component_range, r2_list, rmse_list, best_y_pred, y_test_, best_pca = validate_model(X_test, y_test, component_range, trained_pcas, trained_models, device)
    best_y_pred = np.clip(best_y_pred, None, 1)

    plot_pca(component_range, r2_list, rmse_list)
    plot_scatter(y_test_, best_y_pred, best_pca)
