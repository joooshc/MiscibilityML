import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ast import literal_eval
import ast
import os
from scipy.interpolate import interp1d
import string

myGrey = "#3b4649"
myCyan = "#01e99c"

def phase_overlap(dataset):
    # Base path for saving the plots
    base_save_path = "Results/PlotsForReport/ImprovedPlots/Unused"
    
    # Iterate over all the indices
    for compound_index in range(dataset.shape[0]):
        # Extracting the required data for plotting
        temps = np.array(ast.literal_eval(dataset['Temperatures'][compound_index]))
        true_mole_fracs = np.array(ast.literal_eval(dataset['true_mfs'][compound_index]))
        pred_mole_fracs = np.array(ast.literal_eval(dataset['pred_mfs'][compound_index]))
        compound1_name = dataset['Compound1'][compound_index]
        compound2_name = dataset['Compound2'][compound_index]   
        model_name = dataset['Model'][compound_index]
        r2_value = dataset['R_squared'][compound_index]
        mse_value = dataset['MSE'][compound_index]
        
        plt.figure(figsize=(6, 6), dpi = 300)

        for mfs, color, fill_color, label in [(true_mole_fracs, "black", myGrey, 'True'), (pred_mole_fracs, darkCyan, myCyan, 'Predicted')]:
            # Sorting values based on temperature for plotting
            combined = list(zip(mfs, temps))
            combined.sort(key=lambda x: x[1])
            mfs_sorted, temps_sorted = zip(*combined)

            plt.scatter(mfs_sorted, temps_sorted, color=color, label=label)

            # Getting the smallest and largest mole fraction and its corresponding temperature and drawing a straight line
            x1 = min(mfs_sorted)
            y1 = temps_sorted[mfs_sorted.index(x1)]
            x2 = max(mfs_sorted)
            y2 = temps_sorted[mfs_sorted.index(x2)]
            
            plt.plot([x1, x2], [y1, y2], color=color, linestyle='--')

             # Checking if there are enough points to interpolate
            if len(mfs_sorted) > 3:
                spl = interp1d(mfs_sorted, temps_sorted, kind='linear', fill_value="extrapolate")
                        
                dense_mfs = np.linspace(min(mfs_sorted), max(mfs_sorted), 1000)
                dense_temps_spl = spl(dense_mfs)
                plt.plot(dense_mfs, dense_temps_spl, color=color)

                dense_temps_straight = np.linspace(y1, y2, 1000)
                
                plt.fill_between(dense_mfs, dense_temps_spl, dense_temps_straight, where=(dense_temps_spl < dense_temps_straight), color=fill_color, alpha=0.5)
                plt.fill_between(dense_mfs, dense_temps_spl, dense_temps_straight, where=(dense_temps_spl > dense_temps_straight), color=fill_color, alpha=0.5)

        # Constructing the title and the filename text
        if model_name:
            title_text = f'{compound1_name} & {compound2_name}\nModel: {model_name}\nR²: {r2_value:.4f}, MSE: {mse_value:.4f}'
            filename = f'{compound1_name}_and_{compound2_name}_Model_{model_name}.png'
        else:
            title_text = f'{compound1_name} & {compound2_name}'
            filename = f'{compound1_name}_and_{compound2_name}.png'

        # Remove any unwanted characters from the filename
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        filename = ''.join(c for c in filename if c in valid_chars)
        
        plt.title(title_text, fontsize=18)
        plt.xlabel('Mole Fraction', fontsize=14)
        plt.ylabel('Temperature (°C)', fontsize=14)
        plt.xlim(0.0, 1.0)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(base_save_path, filename)) 
        plt.close()  # Apparently closing the plot frees up memory


df = pd.read_csv("Results/Stats/AllResults_Unique-stats.csv")
myGrey = "#3b4649"
myCyan = "#01e99c"
darkCyan = "#006845"
phase_overlap(df)
# subsets = ["Hexane", "Ethene", "Benzene", "hexane", "ethene", "benzene"]

# for subset in subsets:
#     subset_df = df[df['Compound1'] == subset]
#     subset_df.reset_index(drop=True, inplace=True)
#     phase_overlap(subset_df)
                  