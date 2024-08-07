import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import random, torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from tqdm import tqdm

def quantile_scaler(train_dataset, test_dataset):
    # Sorting the y's
    y_train, y_test = train_dataset['MoleFraction'], test_dataset['MoleFraction']
    train_idx, test_idx = len(y_train)-1, len(y_test)-1

    # Combining the train and test targets into a single vector
    y = pd.concat([y_train, y_test], axis=0)

    scaler = QuantileTransformer(output_distribution = 'normal')

    # Transforming the training targets and test targets jointly so as to keep them within the same quantile distribution (to prevent data leakage)
    scaler.fit(y.values.reshape(-1, 1))
    y_transformed = scaler.transform(y.values.reshape(-1, 1))

    # Reversing back just to test
    y_rvs = scaler.inverse_transform(y_transformed.reshape(-1, 1))

    return y, y_transformed, y_rvs

def plot_hist(y, y_transformed, y_rvs):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.hist(y, bins=30, edgecolor='black')
    plt.title("Original Targets")
    plt.xlabel("Mole Fraction")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.hist(y_transformed, bins=30, edgecolor='black')
    plt.title("Transformed Targets")
    plt.xlabel("Transformed Mole Fraction")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 3)
    plt.hist(y_rvs, bins=30, edgecolor='black')
    plt.title("Reverse Transformed Targets")
    plt.xlabel("Mole Fraction")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def save_to_csv(train_data_path, test_data_path): # Function just to save files 
    # Importing datasets
    train_dataset, test_dataset = pd.read_csv(train_data_path), pd.read_csv(test_data_path)
    train_idx = len(train_dataset)

    # Applying the transform and putting them back into y_train and y_test
    y, y_transformed, y_rvs = quantile_scaler(train_dataset, test_dataset)
    y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]
    train_dataset['MoleFraction'], test_dataset['MoleFraction'] = y_train, y_test

    # Setting export file paths
    train_data_path = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragonQ/QuantileTrain.csv"
    test_data_path = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragonQ/QuantileTest.csv"
    train_dataset.to_csv(train_data_path, index=False)
    test_dataset.to_csv(test_data_path, index=False)

def test_train_scaler_split(train_data_path, test_data_path): # Function to split back to y_train and y_test after scaling

    # Importing datasets
    train_dataset, test_dataset = pd.read_csv(train_data_path), pd.read_csv(test_data_path)
    train_idx = len(train_dataset)

    # Applying the transform and putting them back into y_train and y_test
    y, y_transformed, y_rvs = quantile_scaler(train_dataset, test_dataset)
    y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]
    train_dataset['MoleFraction'], test_dataset['MoleFraction'] = y_train, y_test

    return train_dataset, test_dataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
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

    component_range = range(1, (X.shape[1] + 1), 5)

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
        optimizer = optim.Adam(model.parameters())

        trained_models[n_components] = model.state_dict()

        best_loss = float('inf')
        patience = 500
        no_improve_epochs = 0

        for epoch in range(200):
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

if __name__ == "__main__":
    # Setting import file paths
    train_data_path = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragon/QuantileTrain.csv"
    test_data_path = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragon/QuantileTest.csv"

    # Sort X and y
    train_dataset, test_dataset = test_train_scaler_split(train_data_path, test_data_path)
    X_train, y_train = train_dataset.iloc[:, 7:], train_dataset['MoleFraction']
    X_test, y_test = test_dataset.iloc[:, 7:], test_dataset['MoleFraction']

    # Train and test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    component_range, trained_pcas, trained_models = train_model(X_train, y_train, device)
    component_range, r2_list, rmse_list, best_y_pred, y_test_, best_pca = validate_model(X_test, y_test, component_range, trained_pcas, trained_models, device)
