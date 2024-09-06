import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import ast


#The script reads and processes the results of Random Forest hyperparameter tuning trials, groups them by unique hyperparameter combinations, performs statistical analysis (ANOVA) to see if different hyperparameters lead to significantly different performance, and identifies the combination that gives the best (lowest) score.

# Base directory and subdirectories


predictand = 'mf'
base_dir = f'/glade/campaign/ral/wsap/oracleMLSL/data/hypertuner_output/rf_grid_{predictand}_derivs'
subdirs = [f'noDir_rf_{predictand}_grid_tuner_{i}_hector' for i in range(5)]
#0,"{'n_estimators': 100, 'max_features': 2, 'n_jobs': 2, 'max_leaf_nodes': 32}
# Columns and possible values
n_estimators = [100, 200, 500]
max_features = [2,3,5,10]
max_leaf_nodes= [32,64,128,256,516, 1024]

# Read and combine data
all_data = []
for subdir in subdirs:
    file_path = os.path.join(base_dir, subdir, 'trial_results_.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, names=["TrialID","Hyperparameters","Score","Status"])
        all_data.append(df)


# Concatenate all data
combined_df = pd.concat(all_data, ignore_index=True)

# Function to extract hyperparameters
def extract_hyperparameters(row):
    hyperparameters = ast.literal_eval(row['Hyperparameters'])
    return (
        hyperparameters['n_estimators'],
        hyperparameters['max_features'],
        hyperparameters['max_leaf_nodes'],
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
print(f"n_estimators: {best_combination['hyperparameters'][0]}")
print(f"max_features: {best_combination['hyperparameters'][1]}")
print(f"max_leaf_nodes: {best_combination['hyperparameters'][2]}")
