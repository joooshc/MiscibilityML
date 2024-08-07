import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def r2_mse_bar_chart(file_path1, file_path2):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Unscaled Mole Fractions", "Log Scaled Mole Fractions"]

    for idx, file_path in enumerate([file_path1, file_path2]):
        results_file = pd.read_csv(file_path)
        results_file.rename(columns={'R_squared':'R2'}, inplace=True)

        stats_df = results_file.groupby('Model').agg({'R2': 'max', 'MSE': 'min'}).reset_index()
        stats_df['Root_Model'] = stats_df['Model'].apply(lambda x: x.split('-')[0])
        avg_max_stats_df = stats_df.groupby('Root_Model').agg({'R2': 'mean', 'MSE': 'mean'}).reset_index()

        avg_max_stats_df.sort_values('Root_Model', inplace=True)
        avg_max_stats_df['Root_Model'] = avg_max_stats_df['Root_Model'].replace({'rf': 'RandomForest'})
        avg_max_stats_df['Root_Model'] = avg_max_stats_df['Root_Model'].replace('DNeuralNetwork_LinMF', 'DNN')
        avg_max_stats_df['Root_Model'] = avg_max_stats_df['Root_Model'].replace('DNeuralNetwork', 'DNN')
        avg_max_stats_df['Root_Model'] = avg_max_stats_df['Root_Model'].replace('RandomForest', 'RF')

        x = np.arange(len(avg_max_stats_df['Root_Model']))
        width = 0.35

        ax[0].bar(x + width/2 * idx, avg_max_stats_df['R2'], width, label=labels[idx])
        ax[1].bar(x + width/2 * idx, avg_max_stats_df['MSE'], width, label=labels[idx])
        
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(avg_max_stats_df['Root_Model'])
        a.legend()

    ax[0].set_title('Avg Max R²', fontsize=16)
    ax[0].set_ylabel('R²', fontsize=14)
    ax[0].set_xlabel('ML Algorithm', fontsize=14)
    
    ax[1].set_title('Avg Min MSE', fontsize=16)
    ax[1].set_ylabel('MSE', fontsize=14)
    ax[1].set_xlabel('ML Algorithm', fontsize=14)

    plt.tight_layout()
    plt.show()

def top_20_compound_pairs(file_path1):
    results_file = pd.read_csv(file_path1)
    results_file.rename(columns={'R_squared':'R2'}, inplace=True)
    filtered_results = results_file[results_file['R2'] >= 0.6]
    average_r2 = filtered_results.groupby(['Compound1', 'Compound2'])['R2'].mean().reset_index(name='Average_R2')
    average_mse = filtered_results.groupby(['Compound1', 'Compound2'])['MSE'].mean().reset_index(name='Average_MSE')

    # Grouping by 'Compound1' and 'Compound2', and counting their occurrences
    grouped_counts = filtered_results.groupby(['Compound1', 'Compound2']).size().reset_index(name='Count')
    sorted_results = grouped_counts.sort_values(by='Count', ascending=False)
    sorted_results.to_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/Stats/repeated_compounds-stats.csv")

    # Merging the counts back into the filtered results DataFrame
    filtered_results_with_count = pd.merge(filtered_results, grouped_counts, on=['Compound1', 'Compound2'])

    sorted_results['Compound_Pair'] = sorted_results['Compound1'] + ' & ' + sorted_results['Compound2']
    merged_df = pd.merge(sorted_results, average_r2, on=['Compound1', 'Compound2'])
    final_sorted_results = pd.merge(merged_df, average_mse, on=['Compound1', 'Compound2'])

    print(final_sorted_results)

    plt.figure(figsize=(11, 8))
    plt.barh(final_sorted_results['Compound_Pair'], final_sorted_results['Count'])
    plt.xlabel('Count')
    plt.ylabel('Compound Pairs')
    plt.title('R² > 0.6: Frequency of Compound Pairs')
    plt.show()

def r2_mse_violinplots(file_path):
    results = pd.read_csv(file_path)
    mse = results['MSE']
    r2 = results['R_squared']

    # Filtering out extreme outliers from R^2
    r2 = r2[r2 > -5]
    mse = mse[mse > 0.0] 
    mse = mse[mse < 25]

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Violin plot for R^2
    sns.violinplot(y=r2, ax=axs[0], orient='v')
    axs[0].set_title("R² Distribution", fontsize = 24)
    axs[0].set_ylabel("R² Value", fontsize = 20)
    axs[0].set_xticks([])  
    axs[0].tick_params(axis='y', labelsize=16)

    # Violin plot for MSE
    sns.violinplot(y=mse, ax=axs[1], orient='v')
    axs[1].set_title("MSE Distribution", fontsize = 24)
    axs[1].set_ylabel("MSE Value", fontsize = 20)
    axs[1].set_xticks([])  
    axs[1].tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.show()

def r2_mse_boxplots(file_path):
    results = pd.read_csv(file_path)
    mse = results['MSE']
    r2 = results['R_squared']

    # Filtering out extreme outliers from R^2
    r2_filtered = r2[r2 > -5]
    mse = mse[mse > 0.0]
    mse = mse[mse < 25]

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Boxplot for R^2
    axs[0].boxplot(r2_filtered)
    axs[0].set_title("R² Distribution", fontsize = 24)
    axs[0].set_ylabel("R² Value", fontsize = 20)
    axs[0].set_ylim(min(r2_filtered) - 0.5, max(r2_filtered) + 0.5)  # Setting y-axis limits manually
    axs[0].set_xticks([]) 
    axs[0].tick_params(axis='y', labelsize=16)

    # Boxplot for MSE
    axs[1].boxplot(mse)
    axs[1].set_title("MSE Distribution", fontsize = 24)
    axs[1].set_ylabel("MSE Value", fontsize = 20)
    axs[1].set_xticks([]) 
    axs[1].tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.show()

def compound_r2_bar_chart():
    compound_pairs = [
        "Dibenzothiophene & Tetradecane", "Dodecanoic acid & Methylbenzene", "Hexane & m-Xylene", 
        "Dodecanoic acid & Trichloromethane", "Hexadecanoic acid & Hexane", "Heptadecanoic acid & Ethyl acetate", 
        "Tetradecanoic acid & Ethyl acetate", "Tetrahydrofuran & p-Xylene", "Tetradecanoic acid & Methanol", 
        "2-Propanol & N-Methyl-2-Pyrrolidone", "Dodecanoic acid & Pentan-1-ol", "Tetradecanoic acid & Hexane", 
        "Heptadecanoic acid & Propan-2-ol", "Acenaphthene & 1,1’-Dichloroethane", "Acenaphthene & 1,2-Dichloroethane", 
        "Fluorene & Aniline", "Naphthalene & Tetrachloroethene", "Benzene & p-Xylene", "Dodecanoic acid & Hexadecanoic acid"
    ]

    R2_values = [0.727, 0.728, 0.798, 0.681, 0.755, 0.699, 0.777, 0.720, 0.681, 0.671, 0.637, 0.677, 0.718, 0.738, 0.669, 0.655, 0.760, 0.657, 0.698]
    MSE_values = [0.008, 0.024, 0.017, 0.027, 0.025, 0.032, 0.019, 0.018, 0.025, 0.030, 0.008, 0.029, 0.029, 0.002, 0.002, 0.007, 0.001, 0.026, 0.036]

    fig, axs = plt.subplots(2, 1, figsize=(15, 12))  

# Plotting the R2 values on the first subplot
    axs[0].bar(compound_pairs, R2_values, color='steelblue')
    axs[0].set_ylabel('R²', fontsize=14)
    axs[0].set_title('Avg. R² Scores for the Top 20 Compound Pairs', fontsize=18)
    axs[0].set_xticks([])  

    # Plotting the MSE values on the second subplot
    axs[1].bar(compound_pairs, MSE_values, color='darkgreen')
    axs[1].set_ylabel('MSE', fontsize=14)
    axs[1].set_xlabel('Compound Pairs', fontsize=14)
    axs[1].set_title('Avg. MSE Scores for the Top 20 Compound Pairs', fontsize=18)
    axs[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

file_path1 = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/Stats/AllResults-stats.csv"
file_path2 = "C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Results/Stats/AllResultsY-stats.csv"

# Uncomment to run
'''r2_mse_bar_chart(file_path1, file_path2)'''
'''r2_mse_violinplots(file_path2)'''
'''r2_mse_boxplots(file_path2)'''
compound_r2_bar_chart()