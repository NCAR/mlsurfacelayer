import pandas as pd
import argparse
import sys
import yaml

def calc_mean_metrics_kfold():
    # parser to read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None, help="Config yaml file")
    parser.add_argument("-v", type=int, default=0)
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)

    if args.config == None:
        print("No Yaml file received.")
        sys.exit()

    df_list = []                                                        # Initialize an empty list to store dataframes
    name = config['file_name']
    out_dir = config['out_dir']                                         # outer path to store average metrics csv

    # Loop through the k-folds (0 to 9)
    for num in range(config['k_fold_cross_validation']['N']):
        # Construct the file path
        file_path = f'{out_dir}/{name}/model_QC_--kfold-{num}/surface_layer_model_metrics.csv'
        
        if args.v == 1: print(f'Opening this file: {file_path}')

        df = pd.read_csv(file_path)                                     # Read the CSV file into a dataframe
        
        df['kfold'] = num                                               # Add a column to identify the k-fold
       
        df_list.append(df)                                              # Append the dataframe to the list
   
    if args.v == 1: print(f'Last opened before combining: \n{df}')

    combined_df = pd.concat(df_list, ignore_index=True)                 # Concatenate all dataframes in the list into a single dataframe
    
    if args.v == 1: print(f'columns in combined_df: {combined_df.columns}')
    if args.v >= 2: print(f"Combined_df: \n {combined_df[combined_df['Model'].isin(['heat_flux-mo', 'momentum_flux-mo'])]}")

    average_metrics = combined_df.groupby('Model').mean().reset_index() # Compute the average of the metrics across all folds
    if args.v >= 1: print(f"DF of average metrics: \n {average_metrics[average_metrics['Model'].isin(['heat_flux-mo', 'momentum_flux-mo'])]}")

    average_metrics.to_csv(f'{out_dir}/{name}/average_metrics_{name}.csv', index=False) # Save the result to a new CSV file

if __name__ == "__main__":
    print('starting')
    calc_mean_metrics_kfold()  
    print('ended')








