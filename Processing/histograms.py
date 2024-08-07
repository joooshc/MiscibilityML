import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Datasets/master_unscaled.csv")

def plot_hists(dataset, num_features):
    for feature_window in range(0, len(dataset.columns), num_features):
        features_to_plot = dataset.iloc[:, feature_window:(feature_window)+num_features]
        n_rows = 5
        n_cols = 4

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 2*n_rows)) 
        axes = axes.flatten()

        for idx, feature in enumerate(features_to_plot.columns):
            if idx >= len(axes):  
                break
            ax = axes[idx]
            ax.hist(features_to_plot[feature].dropna(), bins=60)
            ax.set_title(f'{feature}')

        plt.tight_layout()
        plt.savefig(f'Results/Histograms/Master/{feature_window}_raw.png')

dataset = df.iloc[0:, 5:]
plot_hists(dataset, 20)