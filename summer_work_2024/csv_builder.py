'''
This file csv_builder.py is to be copied into a tuning results directory, 
when ran it will access every trial that was ran and compile them into a csv for easier viewing
'''

import os
import json
import pandas as pd

# test function: not working as of july 8 -hector
def testjson():
    # Load oracle.json
    with open('my_dir/hyperparam_min-delta/oracle.json', 'r') as f:
        oracle_data = json.load(f)

    # Explore the structure of the file
    print("Top-level keys in oracle.json:", oracle_data.keys())
    print("Keys in 'hyperparameters':", oracle_data['hyperparameters'].keys())

    # Assuming that the scores might be within 'run_times' or related keys
    if 'run_times' in oracle_data:
        print("Content of 'run_times':", oracle_data['run_times'])

    # Printing hyperparameters
    hyperparameters = oracle_data['hyperparameters']['values']
    print("\nBest Hyperparameters:")
    for param, value in hyperparameters.items():
        print(f"  {param}: {value}")

    # Assuming scores might be under another key, let's explore the structure more
    for key, value in oracle_data.items():
        if isinstance(value, dict):
            print(f"\nExploring '{key}':")
            for sub_key in value.keys():
                print(f"  {sub_key}: {value[sub_key]}")

def main():
    # Define the base directory containing the trial folders
    base_directory = "."

    # List to hold the trial information
    trial_info_list = []

    # Loop through each trial directory (from trial_0 to trial_29)
    for i in range(30):
        trial_folder = f"trial_{i}"
        if i < 10: 
            trial_folder = f"trial_0{i}"
        trial_path = os.path.join(base_directory, trial_folder)
        
        if os.path.isdir(trial_path):
            # Path to the JSON file
            json_file_path = os.path.join(trial_path, 'trial.json')
            
            # Check if the JSON file exists
            if os.path.isfile(json_file_path):
                # Read the JSON file
                with open(json_file_path, 'r') as json_file:
                    trial_data = json.load(json_file)
                    
                    # Extract the relevant information
                    trial_info = {
                        'Trial ID': trial_data.get('trial_id'),
                        'Score': trial_data.get('score'),
                        'Status': trial_data.get('status')
                    }
                    
                    # Add the hyperparameters to the trial_info dictionary
                    hyperparameters = trial_data.get('hyperparameters', {}).get('values', {})
                    trial_info.update(hyperparameters)
                    
                    # Append to the list
                    trial_info_list.append(trial_info)

    # Convert the list to a DataFrame
    tune_res = pd.DataFrame(trial_info_list)

    # Save the DataFrame to a CSV file
    tune_res.to_csv('tuning_results.csv', index=False)

    # Print the DataFrame
    # print(tune_res)

if __name__ == "__main__":
    main()