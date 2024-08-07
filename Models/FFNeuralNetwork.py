import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from sklearn.decomposition import PCA

###### SETUP LOGGING ######
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(message)s')
print = logging.info  # redirect print statements to logging

###### LOAD DATASET ######
# Dragon Flags
DRAGON = False
QUANTILE_D = True
LOG_D = False

# Non Dragon Flags
QUANTILE = True
LOG = False

if DRAGON:
    if QUANTILE_D:
        file_name1 = 'MTrainQDm1'
        file_name2 = 'MTrainQDm2'
        file_name3 = 'MTrainQDm3'
        test_file_name = 'MTestQDm'
    elif LOG_D:
        file_name1 = 'MTrainLDm1'
        file_name2 = 'MTrainLDm2'
        file_name3 = 'MTrainLDm3'
        test_file_name = 'MTestLDm' # MTrain-TestLDm
    else:
        print("Please specify one file name!")
        quit()

    dataset_pt1 = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quantile/{file_name1}.csv")
    dataset_pt2 = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quantile/{file_name2}.csv")
    dataset_pt3 = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Quantile/{file_name3}.csv")
    y1 = dataset_pt1['MoleFraction']
    y2 = dataset_pt2['MoleFraction']
    y3 = dataset_pt3['MoleFraction']
    X1 = dataset_pt1.iloc[:, 7:]
    X2 = dataset_pt2.iloc[:, 7:]
    X3 = dataset_pt3.iloc[:, 7:]

    y = pd.concat([y1, y2, y3], axis=0).reset_index(drop=True)
    X = pd.concat([X1, X2, X3], axis=0).reset_index(drop=True)

else:
    if QUANTILE:
        file_name = 'MTrainQDm'
        test_file_name = 'MTestQDm'
    elif LOG:
        file_name = 'MTrainLDm'
        test_file_name = 'MTestLDm'
    else:
        print("Please specify one file name!")
        quit()

    dataset_pt = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/WithoutDragon/{file_name}.csv")
    
    y = dataset_pt['MoleFraction']
    X = dataset_pt.iloc[:, 7:]

X_weighted = X

# Load the validation data
test_dataset = pd.read_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/WithoutDragon/{test_file_name}.csv")
y_val = test_dataset['MoleFraction']
X_val = test_dataset.iloc[:, 7:]

###### TRAIN ######
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert validation data to PyTorch tensors and move to device
X_val_tensor = torch.tensor(X_val.values).float().to(device)
y_val_tensor = torch.tensor(y_val.values).float().to(device)

# Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Convert data to PyTorch tensors and move to device
X_weighted_tensor = torch.tensor(X_weighted.values).float().to(device)
y_tensor = torch.tensor(y.values).float().to(device)

# Initialize and train the model
model = NeuralNetwork(input_dim=X_weighted_tensor.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = np.int32(np.linspace(100, 5000, 10))
loss_values = [] 
max_components = X.shape[1]
timesteps = 5

# Define a function to compute R-squared
def compute_r2(predictions, true_values):
    ss_res = torch.sum((true_values - predictions) ** 2).item()
    ss_tot = torch.sum((true_values - true_values.mean()) ** 2).item()
    return 1 - (ss_res / ss_tot)

for n_components in range(1, max_components + 1, timesteps):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_val_pca = pca.transform(X_val)
    
    X_tensor_pca = torch.tensor(X_pca).float().to(device)
    X_val_tensor_pca = torch.tensor(X_val_pca).float().to(device)

    # Update model input dimensions based on PCA components
    model = NeuralNetwork(input_dim=n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    print(f"\nTraining with {n_components} PCA components\n")
    
    for epochs in num_epochs:
        pbar = tqdm(total=epochs, desc=f"Training {n_components} components", position=0, leave=True)
        
        for epoch in range(epochs):  # Changed from pbar to range(epochs)
            optimizer.zero_grad()
            outputs = model(X_tensor_pca).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            r2_train = compute_r2(outputs, y_tensor)

            # Calculate R-squared for the validation set
            with torch.no_grad():
                outputs_val = model(X_val_tensor_pca).squeeze()
                r2_val = compute_r2(outputs_val, y_val_tensor) 

            log_details = {"Epoch": epoch, "Train MSE": loss.item(), "Train R2": r2_train, "Test R2": r2_val}
            
            # Update tqdm description with current loss and R-squared values
            pbar.set_postfix(log_details)
            pbar.update(1)

            # Log details only after finishing all epochs in the current batch of num_epochs
            if epoch == epochs - 1:
                logging.info(str(log_details))
                
        pbar.close()