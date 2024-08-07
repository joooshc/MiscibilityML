import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, ast, os, string, time
from ast import literal_eval
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pandas")

def sort_dataframe(dataset, column):
    agg_temps_df = dataset.groupby(['Compound1', 'Compound2', 'SMILES1', 'SMILES2']).agg(Temperatures=('Temperature', list)).reset_index()
    agg_mf_df = dataset.groupby(['Compound1', 'Compound2', 'SMILES1', 'SMILES2']).agg(MoleFractions=(column, list)).reset_index()
    merged_df = pd.merge(agg_temps_df, agg_mf_df, on=['Compound1', 'Compound2', 'SMILES1', 'SMILES2'])

    #print(f"\nagg_temps_df: {agg_temps_df.shape}\nagg_mf_df: {agg_mf_df.shape}\nmerged_df: {merged_df.shape}\n")

    merged_df['Temp_Length'] = merged_df['Temperatures'].apply(len)
    merged_df = merged_df.sort_values(by='Temp_Length', ascending=False).reset_index()
    merged_df.drop(columns=['index'], inplace=True)

    compounds_to_plot = merged_df[merged_df['Temp_Length'] >= 5].reset_index()
    compounds_to_plot.drop(columns=['Temp_Length', 'index'], inplace=True)

    compounds = compounds_to_plot.reset_index(drop=True)

    print(compounds)
    quit()

    return compounds

def interpolate_datapoints(true_compounds, true_temps, true_mfs, min_mfs, max_mfs, model_names, r2_values, mse_values, type_):
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    for i, (temps, mfs) in enumerate(zip(true_temps[:9], true_mfs[:9])):
        ax = axs[i // 3, i % 3]

        combined = list(zip(mfs, temps))
        combined.sort(key=lambda x: x[1])  # Sorting based on the temperature
        mfs_sorted, temps_sorted = zip(*combined)  

        ax.set_xlim(min_mfs, max_mfs)
        ax.scatter(mfs_sorted, temps_sorted, color='black')
        # print(f"mfs_sorted: {mfs_sorted} \ntemps_sorted: {temps_sorted}")

        # Getting the smallest and largest mole fraction and its corresponding temperature and drawing a straight line
        x1 = min(mfs_sorted)
        y1 = temps_sorted[mfs_sorted.index(x1)]
        x2 = max(mfs_sorted)
        y2 = temps_sorted[mfs_sorted.index(x2)]

        ax.plot([x1, x2], [y1, y2], color='black', linestyle='--')

        # Checking if there are enough points in the list for interpolation
        if len(mfs_sorted) > 3:
            spl = interp1d(mfs_sorted, temps_sorted, kind='linear', fill_value="extrapolate")
                    
            # Smoothing the line
            dense_mfs = np.linspace(min(mfs_sorted), max(mfs_sorted), 1000)
            dense_temps_spl = spl(dense_mfs)
            ax.plot(dense_mfs, dense_temps_spl, color='black')
            
            # Define the y-values for the straight line (endpoint line) for the dense x values
            dense_temps_straight = np.linspace(y1, y2, 1000)
            
            # Fill the region between the two curves
            ax.fill_between(dense_mfs, dense_temps_spl, dense_temps_straight, where=(dense_temps_spl < dense_temps_straight), color='lightgrey')
            ax.fill_between(dense_mfs, dense_temps_spl, dense_temps_straight, where=(dense_temps_spl > dense_temps_straight), color='lightgrey')

        compound1_name = true_compounds.iloc[i, 0] 
        compound2_name = true_compounds.iloc[i, 1]  

        model_name = model_names[i]
        r2_value = r2_values[i]
        mse_value = mse_values[i]

        if type_ == 'pred':
            title_text = f'{compound1_name} in {compound2_name}\nModel: {model_name}\nR²: {r2_value:.4f}, MSE: {mse_value:.4f}'
        else:
            title_text = f'{compound1_name} in {compound2_name}'
        ax.set_title(title_text, fontsize=14)
        ax.set_xlabel('Log Scaled Mole Fraction', fontsize=14)
        ax.set_ylabel('Temperature (°C)', fontsize=14)

    for j in range(i+1, 9):
        axs[j // 3, j % 3].axis('off')
        plt.tight_layout()

    plt.tight_layout()
    plt.show()

def extrapolate_datapoints(true_compounds, true_temps, true_mfs, min_mfs, max_mfs):
    pass

def cubic_splines(true_compounds, true_temps, true_mfs):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for i, (temps, mfs) in enumerate(zip(true_temps[:9], true_mfs[:9])):
        ax = axs[i // 3, i % 3]

        temps = np.array(temps)
        mfs = np.array(mfs)

        combined = list(zip(mfs, temps))
        combined.sort(key=lambda x: x[1])  
        mfs_sorted, temps_sorted = zip(*combined)  

        ax.scatter(mfs_sorted, temps_sorted, label='Data Points')
        # print(f"mfs_sorted: {mfs_sorted} \ntemps_sorted: {temps_sorted}")

        # Checking if there are enough points in the list for interpolation
        if len(mfs_sorted) > 3:
            # Interpolating with cubic spline
            spl = interp1d(mfs_sorted, temps_sorted, kind='cubic', fill_value="extrapolate")
            
            # Creaing x values for a smooth line
            dense_mfs = np.linspace(min(mfs_sorted), max(mfs_sorted), 1000)
            ax.plot(dense_mfs, spl(dense_mfs), color='black', label='Cubic Spline')
        
        compound1_name = true_compounds.iloc[i, 0] 
        compound2_name = true_compounds.iloc[i, 1]  
        
        ax.set_title(f'{compound1_name} in {compound2_name}', fontsize=10)
        ax.set_xlabel('MoleFractions', fontsize=10)
        ax.set_ylabel('Temperatures', fontsize=10)

    for j in range(i+1, 9):
        axs[j // 3, j % 3].axis('off')

    plt.tight_layout()
    plt.show()

def rank_top_9(dataset, columns, bottom_worst):
    all_best_nines = []
    acc_over_05 = []
    acc_under_05 = []
    for column in columns:
        true_compounds = sort_dataframe(dataset, f'LogMoleFraction')
        pred_compounds = sort_dataframe(dataset, column)

        true_temps = true_compounds['Temperatures']
        true_mfs = true_compounds['MoleFractions']
        pred_temps = pred_compounds['Temperatures']
        pred_mfs = pred_compounds['MoleFractions']

        r2_values = []
        mse_values = []
        for true_mf, pred_mf in zip(true_mfs, pred_mfs):
            r2 = r2_score(true_mf, pred_mf)
            mse = mean_squared_error(true_mf, pred_mf)
            r2_values.append(r2)
            mse_values.append(mse)

        true_compounds['R_squared'] = r2_values
        pred_compounds['R_squared'] = r2_values
        true_compounds['MSE'] = mse_values
        pred_compounds['MSE'] = mse_values
        sorted_compounds_pred = pred_compounds.sort_values(by='R_squared', ascending=bottom_worst)
        sorted_compounds_true = true_compounds.sort_values(by='R_squared', ascending=bottom_worst)

        def to_sort(sorted_compounds_pred, sorted_compounds_true, top_num=71):
            pred_compounds_sorted = sorted_compounds_pred.iloc[:top_num]
            true_mfs_sorted = sorted_compounds_true['MoleFractions'].iloc[:top_num]
            pred_compounds_sorted['Model'] = column
            pred_compounds_sorted['true_mfs'] = list(true_mfs_sorted) 

            return pred_compounds_sorted

        pred_compounds_sorted = to_sort(sorted_compounds_pred, sorted_compounds_true, 71)
        all_best_nines.append(pred_compounds_sorted)
        
        num_r2_greater_05 = sorted_compounds_pred['R_squared'].values >= 0.6
        num_r2_smaller_05 = sorted_compounds_pred['R_squared'].values <= 0.6

        num=0
        acc_over_05.append(len([num+1 for i in num_r2_greater_05 if i == True]))
        acc_under_05.append(len([num+1 for i in num_r2_greater_05 if i == False]))

    all_best_nines_df = pd.concat(all_best_nines)
    
    # Uncomment to save AllResults-stats.csv
    all_best_nines_df.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/Stats/AllResults-stats.csv", index=False)

    acc_over_05 = np.sum(acc_over_05)
    acc_under_05 = np.sum(acc_under_05)
    print(f"% R² over 0.5: {np.round(100*acc_over_05/(acc_over_05 + acc_under_05), 2)}%")
    print(f"% R² under 0.5: {np.round(100*acc_under_05/(acc_over_05 + acc_under_05), 2)}%")

    added_compounds = set()
    final_best_nine = pd.DataFrame()

    for index, row in all_best_nines_df.sort_values(by='R_squared', ascending=bottom_worst).iterrows():
        compound_pair = tuple(row['true_mfs']) 

        if compound_pair not in added_compounds:
            final_best_nine = final_best_nine.append(row)
            added_compounds.add(compound_pair)

            if len(final_best_nine) >= 9:
                break

    final_best_nine.rename(columns={'MoleFractions':'pred_mfs'}, inplace=True)
    min_mfs = min(min(min(final_best_nine['true_mfs'])), min(min(final_best_nine['pred_mfs'])))
    max_mfs = max(max(max(final_best_nine['true_mfs'])), max(max(final_best_nine['pred_mfs'])))

    model_names = final_best_nine['Model'].tolist()
    r2_values = final_best_nine['R_squared'].tolist()
    mse_values = final_best_nine['MSE'].tolist()

    interpolate_datapoints(final_best_nine, final_best_nine['Temperatures'], final_best_nine['true_mfs'], min_mfs, max_mfs, model_names, r2_values, mse_values, type_='true')
    interpolate_datapoints(final_best_nine, final_best_nine['Temperatures'], final_best_nine['pred_mfs'], min_mfs, max_mfs, model_names, r2_values, mse_values, type_='pred')

def rank_all(dataset):

    # To sort prediction mole fractions in ascending order - CAUTION: Will affect the phase diagram similarity
    '''
    sorted_dataset = dataset.sort_values(by="R_squared", ascending=False)
    unique_dataset = sorted_dataset.drop_duplicates(subset=['Compound1', 'Compound2'], keep='first')

    unique_dataset.rename(columns={'MoleFractions':'pred_mfs'}, inplace=True)

    model_names = unique_dataset['Model'].tolist()
    r2_values = unique_dataset['R_squared'].tolist()
    mse_values = unique_dataset['MSE'].tolist()

    true_temps = unique_dataset['Temperatures']
    true_mfs = unique_dataset['true_mfs']
    pred_mfs = unique_dataset['pred_mfs']

    for i in range(len(true_temps)):
        true_temps.iloc[i] = np.array(literal_eval(true_temps.iloc[i]))
        true_mfs.iloc[i] = np.array(literal_eval(true_mfs.iloc[i]))
        
        # Convert the string to numpy array
        pred_mfs_list = np.array(literal_eval(pred_mfs.iloc[i]))
        
        # Sorting the predicted mole fractions
        sorted_indices_pred = np.argsort(pred_mfs_list)
        pred_mfs_list = pred_mfs_list[sorted_indices_pred]
        
        # Reassign the sorted list to the DataFrame
        pred_mfs.iloc[i] = pred_mfs_list
    
    flat_true_mfs = [item for sublist in true_mfs for item in sublist]
    flat_pred_mfs = [item for sublist in pred_mfs for item in sublist]

    min_mfs = min(min(flat_true_mfs), min(flat_pred_mfs))
    max_mfs = max(max(flat_true_mfs), max(flat_pred_mfs))

    interpolate_datapoints(unique_dataset, true_temps, true_mfs, min_mfs, max_mfs, model_names, r2_values, mse_values, type_='true')
    interpolate_datapoints(unique_dataset, true_temps, pred_mfs, min_mfs, max_mfs, model_names, r2_values, mse_values, type_='pred')

    '''

    sorted_dataset = dataset.sort_values(by="R_squared", ascending=False)
    unique_dataset = sorted_dataset.drop_duplicates(subset=['Compound1', 'Compound2'], keep='first')

    unique_dataset.rename(columns={'MoleFractions':'pred_mfs'}, inplace=True)

    # Uncomment to save the csv file
    unique_dataset.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/Stats/AllResults_Unique-stats.csv", index=False)

    model_names = unique_dataset['Model'].tolist()
    r2_values = unique_dataset['R_squared'].tolist()
    mse_values = unique_dataset['MSE'].tolist()

    true_temps = unique_dataset['Temperatures']
    true_mfs = unique_dataset['true_mfs']
    pred_mfs = unique_dataset['pred_mfs']

    for i in range(len(true_temps)):
        true_temps.iloc[i] = np.array(literal_eval(true_temps.iloc[i]))
        true_mfs.iloc[i] = np.array(literal_eval(true_mfs.iloc[i]))
        pred_mfs.iloc[i] = np.array(literal_eval(pred_mfs.iloc[i]))
    
    flat_true_mfs = [item for sublist in true_mfs for item in sublist]
    flat_pred_mfs = [item for sublist in pred_mfs for item in sublist]

    min_mfs = min(min(flat_true_mfs), min(flat_pred_mfs))
    max_mfs = max(max(flat_true_mfs), max(flat_pred_mfs))

    interpolate_datapoints(unique_dataset, true_temps, true_mfs, min_mfs, max_mfs, model_names, r2_values, mse_values, type_='true')
    interpolate_datapoints(unique_dataset, true_temps, pred_mfs, min_mfs, max_mfs, model_names, r2_values, mse_values, type_='pred')

def phase_overlap(dataset):
    # Base path for saving the plots
    base_save_path = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/PlotsForReport/Overlapping Phase Diagrams"
    
    # Iterate over all the indices
    for compound_index in range(71):
        # Extracting the required data for plotting
        temps = np.array(ast.literal_eval(dataset['Temperatures'][compound_index]))
        true_mole_fracs = np.array(ast.literal_eval(dataset['true_mfs'][compound_index]))
        pred_mole_fracs = np.array(ast.literal_eval(dataset['pred_mfs'][compound_index]))
        compound1_name = dataset['Compound1'][compound_index]
        compound2_name = dataset['Compound2'][compound_index]   
        model_name = dataset['Model'][compound_index]
        r2_value = dataset['R_squared'][compound_index]
        mse_value = dataset['MSE'][compound_index]
        
        plt.figure(figsize=(6, 6))

        for mfs, color, fill_color, label in [(true_mole_fracs, 'darkred', 'lightcoral', 'True'), (pred_mole_fracs, 'navy', 'lightskyblue', 'Predicted')]:
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

results_dataset = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/NoDragon/LogTrain.csv")
stats_dataset = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/Stats/AllResults-stats.csv")
stats_dataset_unq = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/Stats/AllResults_Unique-stats.csv")
columns = results_dataset.iloc[:, 9:].columns

# Uncomment to rank all
'''rank_all(stats_dataset)'''

# Uncomment to plot the phase overlap
phase_overlap(stats_dataset_unq)

# Uncomment to rank top 9
'''rank_top_9(results_dataset, columns, bottom_worst=False)'''
