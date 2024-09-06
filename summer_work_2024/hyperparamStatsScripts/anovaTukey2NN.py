import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import ast
import networkx as nx

#The code reads in neural network hyperparameter tuning results, extracts key hyperparameters, groups and analyzes them, and performs statistical tests (ANOVA and Tukey's HSD) to compare the performance of different hyperparameter combinations.
#The data is grouped by combinations of hyperparameters, and the mean score and count of trials for each combination are calculated.
#Combinations with at least 5 observations are kept for further analysis in the filtered_grouped dataframe.
#The best set of hyperparameters is determined by finding the combination with the lowest mean score.
#The code prepares the data for ANOVA by gathering the score data for each group of hyperparameters.
#If there are more than one group with sufficient observations, an ANOVA test is performed to check if the differences between groups are statistically significant.
#If the ANOVA test is significant (p-value < 0.05), Tukey's HSD test is performed to check which specific groups of hyperparameters are statistically different from each other.
#The best set of hyperparameters is compared with other sets, and the code identifies which combinations are statistically different and which are the same.
#The results of the Tukey test are printed, including how many sets are statistically different from the best set and how many are statistically the same.

predictand = 'mf'
# Base directory and subdirectories
base_dir = '/glade/campaign/ral/wsap/oracleMLSL/data/hypertuner_output'
subdirs = [f'noDir_nn_{predictand}_grid_tuner_{i}_hector' for i in range(5)]

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
    activation = hyperparameters['activation']
    if activation == 'relu':
        activation = 1
    elif activation == 'tanh':
        activation = 0
    return (
        hyperparameters['hidden_layers'],
        hyperparameters['hidden_neurons'],
        hyperparameters['learning_rate'],
        activation
    )

# Extract hyperparameters and add them as separate columns
combined_df[['hidden_layers', 'hidden_neurons', 'learning_rate', 'activation']] = combined_df.apply(
    lambda row: pd.Series(extract_hyperparameters(row)), axis=1
)

# Group by hyperparameter combinations and calculate mean score and count
grouped = combined_df.groupby(['hidden_layers', 'hidden_neurons', 'learning_rate', 'activation']).agg(
    mean_score=('Score', 'mean'),
    count=('Score', 'count')
).reset_index()

# Filter combinations with at least 5 observations
min_observations = 5
filtered_grouped = grouped[grouped['count'] >= min_observations]

# Find the best set of hyperparameters with the lowest mean score
best_hyperparameters = filtered_grouped.loc[filtered_grouped['mean_score'].idxmin()]

# Prepare data for ANOVA
anova_data = []
for _, group in filtered_grouped.iterrows():
    scores = combined_df[
        (combined_df['hidden_layers'] == group['hidden_layers']) &
        (combined_df['hidden_neurons'] == group['hidden_neurons']) &
        (combined_df['learning_rate'] == group['learning_rate']) &
        (combined_df['activation'] == group['activation'])
    ]['Score']
    anova_data.append(scores)

# Perform ANOVA
if len(anova_data) > 1:
    anova_result = stats.f_oneway(*anova_data)
    print(f"ANOVA F-statistic: {anova_result.statistic}")
    print(f"ANOVA P-value: {anova_result.pvalue}")

    if anova_result.pvalue < 0.05:
        print("ANOVA is significant. Performing Tukey's HSD test...")
        # Prepare data for Tukey HSD test
        combined_df['hyperparameter_combination'] = combined_df.apply(
            lambda row: (row['hidden_layers'], row['hidden_neurons'], row['learning_rate'], row['activation']), axis=1
        )
        groups=combined_df['hyperparameter_combination'] 
        tukey_result = pairwise_tukeyhsd(endog=combined_df['Score'], groups=combined_df['hyperparameter_combination'], alpha=0.05)
        tukey_summary = tukey_result.summary()
        print(tukey_summary)
        
        # Summarize Tukey HSD results
        summary_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
        significant_results = summary_df[summary_df['reject'] == True]
        
        # Check which sets are statistically different from the best set
        best_set = tuple(best_hyperparameters[['hidden_layers', 'hidden_neurons', 'learning_rate', 'activation']])
        print(best_set)
        
        statistically_different = significant_results[
            (significant_results['group1'] == best_set) | (significant_results['group2'] == best_set)
        ]
        
        # Check which sets are statistically the same as the best set
        statistically_same = summary_df[
            ~summary_df.index.isin(statistically_different.index)
        ]
        
        print("\nStatistically different results from the best set of hyperparameters:")
        print("there are ", len(statistically_different.index))
        print(statistically_different)
        
        print("\nStatistically same results as the best set of hyperparameters:")
        print("there are ", len(statistically_same.index))
        print(statistically_same)

else:
    print("Not enough groups with sufficient observations to perform ANOVA")

