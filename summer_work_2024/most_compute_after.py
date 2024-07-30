#!/usr/bin/env python
import numpy as np
import math
import yaml
import argparse
import pandas as pd 
#from mlsurfacelayer.data import load_derived_data
from mlsurfacelayer.mvco_data_QC import load_derived_data_random_test_train
from mlsurfacelayer.mvco_data_QC import load_derived_data_random_test_train_val
from mlsurfacelayer.models import save_random_forest_csv, save_scaler_csv
from mlsurfacelayer.mo import * 

from mlsurfacelayer.mo import computeMOSTfluxes

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from mlsurfacelayer.models import DenseNeuralNetwork
from mlsurfacelayer.explain import feature_importance
from os import makedirs
from os.path import exists, join
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlsurfacelayer.metrics import mean_error, hellinger_distance, pearson_r2
import pickle
from keras.models import save_model

model_classes = {"random_forest": RandomForestRegressor,
                 "neural_network": DenseNeuralNetwork}

metrics = {"mean_squared_error": mean_squared_error,
           "mean_absolute_error": mean_absolute_error,
           "pearson_r2": pearson_r2,
           "hellinger_distance": hellinger_distance,
           "mean_error": mean_error}

def comp_(exp_name):
    df_list = []

    with open(f'../config/{exp_name}.yml', "r") as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)

    for i in range(config['k_fold_cross_validation']['N']):
    # for i in range(2):
        with open(f'../../data/model_output/{exp_name}/{exp_name}--kfold-{i}.yml', "r") as config_file:
            config = yaml.load(config_file,Loader=yaml.FullLoader)
        
        # split data into fold, use the associated yml file
        # data = load_derived_data_random_test_train_val(config['data_file'], dropna=True, devVar=config['k_fold_cross_validation']['devVar'],N=config['k_fold_cross_validation']['N'],k=config['k_fold_cross_validation']['k'],holdout_ratio=config['k_fold_cross_validation']['holdout_ratio'],scramble=config['k_fold_cross_validation']['scramble'],config=config)
        # for a in data: # iterates through the train, val, & test, sets to apply the change
        #     data[a]['momentum_flux:18.4_m:m2_s-2'] = data[a]['momentum_flux:18.4_m:m2_s-2'].replace(-0.0, 0)

        data = pd.read_csv(f'../../data/model_output/{exp_name}/model_QC_--kfold-{i}/surface_layer_model_predictions.csv')

        model_metrics = pd.DataFrame(0, index=['momentum_flux-mo', 'heat_flux-mo'], columns=config["model_metric_types"], dtype=np.float32)
        
        for output_type in config['output_columns']:
            mo_predictand_label = output_type + "-" + "mo"

            for model_metric in config["model_metric_types"]:
                # valid_indices = ~np.isnan(data["test"][config["output_columns"][output_type]].values) & ~np.isnan(data['test']['MOST_' + config["output_columns"][output_type].replace('C','K')].values)
                valid_indices = ~np.isnan(data[config["output_columns"][output_type]].values) & ~np.isnan(data['MOST_' + config["output_columns"][output_type].replace('C','K')].values)

                # model_metrics.loc[mo_predictand_label,model_metric] = metrics[model_metric](data["test"][config["output_columns"][output_type]].values[valid_indices], data['test']['MOST_' + config["output_columns"][output_type].replace('C','K')].values[valid_indices])
                model_metrics.loc[mo_predictand_label,model_metric] = metrics[model_metric](data[config["output_columns"][output_type]].values[valid_indices], data['MOST_' + config["output_columns"][output_type].replace('C','K')].values[valid_indices])
        # add line here that adds to the end of the metrics another columns called "nan_count", will show how any nans where in the MOST section
        model_metrics['nan_count'] = len(~np.isnan(data['MOST_' + config["output_columns"][output_type].replace('C','K')].values))
        model_metrics['nan_count2'] = len(~np.isnan(data[config["output_columns"][output_type]].values))
        model_metrics['total_count'] = len(data)
        
        model_metrics.to_csv(f'../../data/model_output/{exp_name}/model_QC_--kfold-{i}/most_metrics.csv', index_label="Model")
        df = pd.read_csv(f'../../data/model_output/{exp_name}/model_QC_--kfold-{i}/most_metrics.csv')
        df['kfold'] = i                                               # Add a column to identify the k-fold
        df_list.append(df)  
    
    print(model_metrics, 'last opened before cobining')
    
    combined_df = pd.concat(df_list, ignore_index=True)                 # Concatenate all dataframes in the list into a single dataframe

    average_metrics = combined_df.groupby('Model').mean().reset_index() # Compute the average of the metrics across all folds

    average_metrics.to_csv(f'../../data/model_output/{exp_name}/average_MOST_metrics_{exp_name}.csv', index=False) # Save the result to a new CSV file
    print(average_metrics)


if __name__ == "__main__":
    # comp_(exp_name='exp_derivs_sfcVars')
    # comp_(exp_name='exp_heightDependent')
    comp_(exp_name='exp4')