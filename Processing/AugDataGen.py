import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon

def plot_hist(y, distribution):
    plt.hist(y, bins=60, edgecolor='black')
    plt.title(f'Mole Fractions - {distribution}')
    plt.xlabel('Mole Fractions')
    plt.ylabel('Frequency')
    plt.show()

def uniform(combined):
    num_synthetic_samples = int(10 * combined.shape[0])

    # Generating synthetic samples for X
    synthetic_X = pd.DataFrame()
    for column in X.columns:
        min_val = X[column].min()
        max_val = X[column].max()
        synthetic_values = np.random.uniform(min_val, max_val, num_synthetic_samples)
        synthetic_X[column] = synthetic_values

    # Generating synthetic y values
    synthetic_y = np.random.uniform((y.min() + 0.01), y.max(), num_synthetic_samples)
    synthetic_combined = pd.concat([pd.Series(synthetic_y, name='MoleFraction'), synthetic_X], axis=1)

    # Concatenating them back
    augmented_combined = pd.concat([combined, synthetic_combined], ignore_index=True)
    print(combined.shape, augmented_combined.shape)
    plot_hist(augmented_combined['MoleFraction'])

def custom_distribution(p):
    peak_position = 0.1
    a = 10 # controls the height of the peak
    b = 10 # controls the spread of the peak
    c = 2 # controls the rate of decay
    return a * np.exp(-b * (p - peak_position) ** 2) * np.exp(-c * p)

def poisson(combined, X):
    num_synthetic_samples = int(2 * combined.shape[0])
    synthetic_y = []
    for _ in range(num_synthetic_samples):
        while True:
            p = np.random.uniform(0, 1)
            q = np.random.uniform(0, custom_distribution(0.1))
            if q < custom_distribution(p):
                synthetic_y.append(p)
                break

    synthetic_X = pd.DataFrame()
    for column in X.columns:
        min_val = X[column].min()
        max_val = X[column].max()
        synthetic_values = np.random.uniform(min_val, max_val, num_synthetic_samples)
        synthetic_X[column] = synthetic_values

    synthetic_combined = pd.concat([pd.Series(synthetic_y, name='MoleFraction'), synthetic_X], axis=1)
    augmented_combined = pd.concat([combined, synthetic_combined], ignore_index=True)
    print(combined.shape, augmented_combined.shape)
    plot_hist(augmented_combined['MoleFraction'])

def gamma(combined, X):
    num_synthetic_samples = int(2 * combined.shape[0])
    scale = 0.1  # decay rate

    synthetic_y = expon.rvs(scale=scale, size=num_synthetic_samples)
    synthetic_y = synthetic_y[(synthetic_y < 1) & (synthetic_y >= 0.02)]  # Ensuring values are in the 0.05 to 1 range

    # Randomly select existing samples from X
    synthetic_X = X.sample(n=len(synthetic_y), replace=True).reset_index(drop=True)

    synthetic_combined = pd.concat([pd.Series(synthetic_y, name='MoleFraction'), synthetic_X], axis=1)
    augmented_combined = pd.concat([combined, synthetic_combined], ignore_index=True)
    print(combined.shape, augmented_combined.shape)
    plot_hist(augmented_combined['MoleFraction'], 'Gamma')

    return augmented_combined

def create_synthetic_metadata(n_rows):
    compound1 = ["C1_n" + str(i) for i in range(n_rows)]
    compound2 = ["C2_n" + str(i) for i in range(n_rows)]
    smiles1 = ["SM_n" + str(i) for i in range(n_rows)]
    smiles2 = ["SM_n" + str(i) for i in range(n_rows)]
    
    synthetic_metadata = pd.DataFrame({
        'Compound1': compound1,
        'Compound2': compound2,
        'SMILES1': smiles1,
        'SMILES2': smiles2
    })

    return synthetic_metadata

def split_dragon_files(file_path):
    dataset = pd.read_csv(file_path)

    split_idx = len(dataset) // 2

    part1 = dataset.iloc[:split_idx]
    part2 = dataset.iloc[split_idx:]

    base_name = file_path.split('/')[-1].split('.')[0]
    new_name1 = base_name + '_1.csv'
    new_name2 = base_name + '_2.csv'

    part1.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Augmented/Gamma/Dragon/{new_name1}", index=False)
    part2.to_csv(f"C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Augmented/Gamma/Dragon/{new_name2}", index=False)

    print(f'{file_path} has been split into {new_name1} and {new_name2}')

dataset = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragon/LogTrain.csv")
y = dataset['MoleFraction']
X = dataset.iloc[:, 7:]
combined = pd.concat([y, X], axis=1)
augmented =  gamma(combined, X)

metadata = dataset.iloc[:, :4]
synthetic_metadata = create_synthetic_metadata(augmented.shape[0] - dataset.shape[0])
augmented_metadata = pd.concat([metadata, synthetic_metadata], ignore_index=True)
final_dataset = pd.concat([augmented_metadata, augmented], axis=1)
print(synthetic_metadata.shape)
print(augmented_metadata.shape)
print(final_dataset.shape)
# final_dataset.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Augmented/Gamma/Dragon/LogTrain1.csv", index=False)
# split_dragon_files("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Augmented/Gamma/Dragon/LogTrain1.csv")