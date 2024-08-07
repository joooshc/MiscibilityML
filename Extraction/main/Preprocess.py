import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, json, os

class Preprocessing:

    @staticmethod
    def log_scaling(dataset):
        dataset_scaled = dataset.copy()

        for col in dataset_scaled.columns:
            min_val = dataset_scaled[col].min()
            shift = 0
            if min_val <= 0:
                shift = -min_val + 1e-7  
            dataset_scaled[col] = np.log(dataset_scaled[col] + shift)

        dataset_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset_scaled.fillna(dataset_scaled.mean(), inplace=True)

        return dataset_scaled, dataset_scaled.iloc[:, 0]

    @staticmethod
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
            plt.show()

    @staticmethod
    def check_skewness(dataset):
        skewness_index = []
        skewness = []
        for col in dataset.columns:
            print(f'Skewness of {col}: {dataset[col].skew()}')
            skewness_index.append((col, dataset[col].skew()))
            skewness.append(dataset[col].skew())
        print(f"\n\nAverage Skewness: {np.mean(skewness)}")

    @staticmethod
    def drop_single_value_columns(dataset):
        # This function checks each column and drops the ones where all values are the same
        for col in dataset.columns:
            if len(dataset[col].unique()) == 1:
                print(f'Dropped {col}')
                dataset.drop(col, inplace=True, axis=1)
        return dataset
    
    @staticmethod
    def load_json(json_filepath):
        with open(json_filepath) as json_file:
            data = json.load(json_file)
        return data['to remove']

    @staticmethod
    def process_dataset(dataset, num_features, plot_hist, plot_scaled_hist, root_directory):
        warnings.filterwarnings("ignore", category=RuntimeWarning) 

        # Build the path to the FeatureEditing.json file dynamically
        json_filepath = os.path.join(root_directory, "Datasets", "IUPAC", "FeatureEditing.json")
        cols_to_remove = Preprocessing.load_json(json_filepath)
        dataset = dataset.drop(cols_to_remove, axis=1, errors='ignore')

        dataset = Preprocessing.drop_single_value_columns(dataset)
        _, y_scaled = Preprocessing.log_scaling(dataset)

        if plot_hist and not plot_scaled_hist:  # to plot histogram before scaling
            Preprocessing.check_skewness(dataset)
            Preprocessing.plot_hists(dataset, num_features)

        dataset, _ = Preprocessing.log_scaling(dataset)

        if plot_hist and plot_scaled_hist:  # to plot histogram after scaling
            Preprocessing.check_skewness(dataset)
            Preprocessing.plot_hists(dataset, num_features)

        print(pd.DataFrame(dataset).shape)
        dataset['MoleFraction'] = y_scaled
        
        return dataset
