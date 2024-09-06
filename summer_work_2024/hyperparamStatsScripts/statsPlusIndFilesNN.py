import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import ast

#the script reads and processes the results of NN hyperparameter tuning trials, groups them by unique hyperparameter combinations, performs statistical analysis (ANOVA) to see if different hyperparameters lead to significantly different performance, and identifies the combination that gives the best (lowest) score.

# Base directory and subdirectories
base_dir = '/glade/campaign/ral/wsap/oracleMLSL/data/hypertuner_output'
subdirs = [f'noDir_nn_mf_grid_tuner_{i}_hector' for i in range(5)]

# Columns and possible values
hidden_layer_vals = [1, 3, 5]
hidden_neurons_vals = [16, 32, 64, 128]
learning_rate_vals = [0.01, 0.001, 0.0001]
activation_vals = ["'relu'", "'tanh'"]

# Read and combine data
all_data = []
for subdir in subdirs:
    file_path = os.path.join(base_dir, subdir, 'combined_trial_results4.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        all_data.append(df)

# Concatenate all data
combined_df = pd.concat(all_data, ignore_index=True)

# Function to extract hyperparameters
def extract_hyperparameters(row):
    hyperparameters = ast.literal_eval(row['Hyperparameters'])
    return (
        hyperparameters['hidden_layers'],
        hyperparameters['hidden_neurons'],
        hyperparameters['learning_rate'],
        hyperparameters['activation']
    )

# Extract hyperparameters
combined_df['hyperparameters'] = combined_df.apply(extract_hyperparameters, axis=1)

# Group by hyperparameter combinations and calculate mean score and count
grouped = combined_df.groupby('hyperparameters').agg(
    mean_score=('Score', 'mean'),
    count=('Score', 'count')
).reset_index()

# Filter combinations with at least 5 observations
min_observations = 5
filtered_grouped = grouped[grouped['count'] >= min_observations]

# Create score arrays for each combination
score_arrays = [
    combined_df[combined_df['hyperparameters'] == hp]['Score'].values
    for hp in filtered_grouped['hyperparameters']
]

# Perform ANOVA if there are sufficient groups with multiple observations
if len(score_arrays) > 1:
    anova_result = stats.f_oneway(*score_arrays)
    print(f"ANOVA F-statistic: {anova_result.statistic}")
    print(f"ANOVA P-value: {anova_result.pvalue}")
else:
    print("Not enough groups with sufficient observations to perform ANOVA")

# Identify the combination with the lowest mean score
best_combination_idx = filtered_grouped['mean_score'].idxmin()
best_combination = filtered_grouped.iloc[best_combination_idx]

print("Best Hyperparameter Combination:")
print(best_combination)

# Additional output for clarity
print("\nHyperparameters leading to the best score:")
print(f"Hidden Layers: {best_combination['hyperparameters'][0]}")
print(f"Hidden Neurons: {best_combination['hyperparameters'][1]}")
print(f"Learning Rate: {best_combination['hyperparameters'][2]}")
print(f"Activation: {best_combination['hyperparameters'][3]}")
