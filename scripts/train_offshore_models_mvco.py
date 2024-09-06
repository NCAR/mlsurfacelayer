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

def main():
    #
    # Parse program args:  config file path 
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)

    #
    # Training data
    #
    data_file = config["data_file"]

    #
    # Output directory for ML models and performance statistics
    #
    out_dir = config["out_dir"]

    #
    # The prediction labels
    #
    output_types = config["output_types"]
    print("ML prediction types: ", output_types)

    #
    # The predictors (column labels)
    #
    input_columns = config["input_columns"]

    #
    # The predictand ( column labels)
    #
    output_columns = config["output_columns"]
    output_col_names = [x for x in sorted(list(output_columns.values()))]

    #
    # Variables to output with predictions
    #
    derived_columns = config["derived_columns"]

    #
    # ML model definitions and parameters
    #
    model_configs = config["model_config"]

    #
    # Error metrics
    #
    model_metric_types = config["model_metric_types"]

    #
    # Predictor importance tests for the resulting models will be performed 
    # for three stability regimes (stable, neutral unstable)
    # stability column is the label for variable used for separating data by regime
    #
    stability_column = config["stability_column"]

    #
    # Create the output directory for models and results if needed
    #
    if not exists(out_dir):
        makedirs(out_dir)

    #
    # Test and train data loaded.
    # Testing data specified by randomly chosen week within each month of data
    #
    data = load_derived_data_random_test_train_val(data_file, 
                                                   devVar=config['k_fold_cross_validation']['devVar'],
                                                   N=config['k_fold_cross_validation']['N'],
                                                   k=config['k_fold_cross_validation']['k'],
                                                   holdout_ratio=config['k_fold_cross_validation']['holdout_ratio'],
                                                   scramble=config['k_fold_cross_validation']['scramble'],
                                                   config=config)#,
    #
    # Output the train, validate, test, holdout sets
    #
    for a in data:
        data[a].to_csv(join(out_dir, f'data_from_{a}_split.csv'), index_label='Time')

    #
    # Create a data frame for the predictions and the derived or predictor variables
    # to be included with the predictions. Output will be in surface_layer_model_predictions.csv 
    #
    pred_columns = []
    # from yaml config sample output_types: ['momentum_flux', 'heat_flux']
    for output_type in output_types:
        #
        # Predictand names will be "<modelName>-<predictand>"
        # 
        for model_name in model_configs.keys():
            pred_columns.append(output_type + "-" + model_name)
        
        #
        # Predictands computed using Monin Obukhov Similarity Theory
        #
        pred_columns.append(output_type + "-" + "mo")

    #
    # Initialize the model prediction data frame
    #
    model_predictions = pd.DataFrame(0, index=data["test"].index,
                                     columns=pred_columns + derived_columns + output_col_names,
                                     dtype=np.float32)
    #
    # Copy in the derived data columns from the test dataset
    #
    print( "predictions index: ", len(model_predictions.index), " data[test].index:",  len(data["test"].index))
    

    print(data["test"].columns)
    print(derived_columns)

    tempvar = data["test"][derived_columns]
    model_predictions.loc[:, derived_columns] = tempvar

    #
    # Create a data frame with predictands as index and columns with metric label. 
    # Data frame will contain average error using each predictive or estimation method
    #
    model_metrics = pd.DataFrame(0, index=pred_columns, columns=model_metric_types,
                                 dtype=np.float32)
    print(model_metrics)
    
    # Save the YAML config used for this model to the results folder (allows us to look at the params used)
    with open(out_dir + '/config_info.yml', 'w') as file:
        yaml.dump(config, file) 

    #
    # Compute the error in the MOST estimations of the predictands 
    # MOST estimates are computed along with other derived data values in mlsurfacelayer lib files 
    # and fill in the model_metrics dataframe with the results
    # from yaml config sample output_types: ['momentum_flux', 'heat_flux']
    for output_type in output_types:
        # Get the predictand label defined above which is used in the model_metrics dataframe
        mo_predictand_label = output_type + "-" + "mo"

        # Compute the different error metrics for each mo estimated predictand 
        for model_metric in model_metric_types:
            # Ensure there are no NaNs in your predictions
            # (If heat flux has units in degrees C, replace with degrees K)
            valid_indices = ~np.isnan(data["test"][output_columns[output_type]].values) & ~np.isnan(data['test']['MOST_' + output_columns[output_type].replace('C','K')].values)
            
            # Execute the metric function and for this mo index label and metric column
            # fill in the model_metrics data frame  
            model_metrics.loc[mo_predictand_label,model_metric] = metrics[model_metric](data["test"][output_columns[output_type]].values[valid_indices], data['test']['MOST_' + output_columns[output_type].replace('C','K')].values[valid_indices])


    #
    # Create ML models 
    #
    model_objects = dict()

    #
    # Dictionary of data normalizer values will contain mean and std for each scaled predictor
    #
    input_scalers = {}

    #
    # Dictionary of predictor importances
    #
    importances = {}

    #
    # for each prediction problem/label or predictand
    #
    print(model_predictions.columns)

    for output_type in output_types:
        print("\n\n")
        print("Predictand: " ,output_columns[output_type])

        print("train predictors shape: ",data["train"][input_columns[output_type]].shape)
        print("train predictand shape: ",data["train"][output_columns[output_type]].shape)
        print("test predictand shape: ", data["test"][output_columns[output_type]].shape)

        #
        # Copy the test truth data for this predictand to the model_predictions data frame
        #
        model_predictions.loc[:, output_columns[output_type]] = data["test"][output_columns[output_type]]
        
        #
        # Data normalizer
        #
        input_scalers[output_type] = StandardScaler()
        
        importances[output_type] = {}

        #
        # Scale training and validation data for ANN using the normalizer (predictors and predictand)
        #
        var_scale_list = input_columns[output_type] + [output_columns[output_type]]

        print(var_scale_list)
        scaled_train  = input_scalers[output_type].fit_transform( data["train"][var_scale_list])
        scaled_val  = input_scalers[output_type].transform( data["validate"][var_scale_list])
        
        #
        # Scale the test data predictors
        #
        scaled_test = input_scalers[output_type].transform(data["test"][var_scale_list])

        #
        # Train the models for this prediction problem/label
        #
        for model_name, model_config in model_configs.items():
            
            #
            # Model container/dictionary
            #
            model_objects[model_name] = {}
            print("Training", output_type, model_name)
            
            #
            # Instantiate the models with the specified parameter in the config file
            #
            model_objects[model_name][output_type] = model_classes[model_name](**model_config)

            print("The predictors ", input_columns[output_type])
            print("Training data shape: ",data["train"][input_columns[output_type]].shape)

            #
            # Train the random forest on non-normalized data
            #
            if model_name == "random_forest":
                model_objects[model_name][output_type].fit(data["train"][input_columns[output_type]].values,
                                                           data["train"][output_columns[output_type]].values)
            #
            # Train the neural net on the normalized data
            #
            else:
                history = model_objects[model_name][output_type].fit(scaled_train[:,0:-1], scaled_train[:,-1], scaled_val[:,0:-1], scaled_val[:,-1])
                # print(history, history.history)
            
            print("\nPredicting", output_type, model_name)
            
            predictandLabel_model = output_type + "-" + model_name

            #
            # Separate data by stability regime
            #
            unstable = data["train"][stability_column] < -0.02
            stable = data["train"][stability_column] > 0.02
            neutral = (data["train"][stability_column] >= -0.02) & (data["train"][stability_column] <= 0.02)

            if model_name == "random_forest":
                #
                # Run random forest on test data, fill in the model_predictions column with output
                #
                model_predictions.loc[:, predictandLabel_model] = model_objects[model_name][output_type].predict(data["test"][input_columns[output_type]])
                print(" random forest prediction min", predictandLabel_model, ": ", model_predictions.loc[:, predictandLabel_model].min())

                #
                # Compute feature importances for all stability regimes
                #
                print("Computing RF predictor importance tests for all stability regimes")
                importances[output_type][model_name] = feature_importance(
                    data["train"][input_columns[output_type]].values,
                    data["train"][output_columns[output_type]].values,
                    model_objects[model_name][output_type],
                    mean_absolute_error,
                    x_columns=input_columns[output_type],
                    col_start="all_")


                #
                # Compute feature importances for neutral regime
                #
                #print("Computing RF predictor importance tests for stability regimes")
                #importances[output_type][model_name] = feature_importance(
                #    data["train"].loc[neutral, input_columns[output_type]].values,
                #    data["train"].loc[neutral, output_columns[output_type]].values,
                #    model_objects[model_name][output_type],
                #    mean_squared_error,
                #    x_columns=input_columns[output_type],
                #    col_start="neutral_")
                #
                # Compute feature importances for unstable regime 
                #
                #importances[output_type][model_name] = pd.concat([feature_importance(
                #    data["train"].loc[unstable, input_columns[output_type]].values,
                #    data["train"].loc[unstable, output_columns[output_type]].values,
                #    model_objects[model_name][output_type],
                #    mean_squared_error,
                #    x_columns=input_columns[output_type],
                #    col_start="unstable_"), importances[output_type][model_name]], axis=1)

                #
                # Compute feature importances for stable regime
                #
                #importances[output_type][model_name] = pd.concat([feature_importance(
                #    data["train"].loc[stable, input_columns[output_type]].values,
                #    data["train"].loc[stable, output_columns[output_type]].values,
                #    model_objects[model_name][output_type],
                #    mean_squared_error,
                #    x_columns=input_columns[output_type],
                #    col_start="stable_"), importances[output_type][model_name]], axis=1)

                #
                # output importance data to csv file
                #
                importances[output_type][model_name].to_csv(join(out_dir,
                                                                output_type + "_" + model_name + "_importances.csv"),
                                                                index_label="input")
            else:
                #
                # Run neural network on scaled test data, fill in the model_predictions column with output
                #
                model_predictions.loc[:, predictandLabel_model] = model_objects[
                    model_name][output_type].predict(scaled_test[:,0:-1] )

                # Check for NaNs
                def check_for_nans(data):
                    if isinstance(data, pd.DataFrame):
                        nan_summary = data.isna().sum()
                        print("NaNs in DataFrame:", nan_summary)
                    elif isinstance(data, np.ndarray):
                        nan_summary = np.isnan(data).sum()
                        print("NaNs in ndarray:", nan_summary)
                    else:
                        print("Unsupported data type")

                # Example usage with your scaled_train DataFrame
                check_for_nans(scaled_train[:,0:-1])
                check_for_nans(scaled_train[:,-1])

                #
                # Compute feature importances for neutral regime 
                #
                print("Computing Neural Network predictor importance tests for stability regimes")
                importances[output_type][model_name] = feature_importance(scaled_train[:,0:-1],
                                                                            scaled_train[:,-1],
                                                                            model_objects[model_name][output_type],
                                                                            mean_absolute_error,
                                                                            x_columns=input_columns[output_type],
                                                                            col_start="all_")
                
                #
                # Compute feature importances for neutral regime 
                #
                #print("Computing Neural Network predictor importance tests for stability regimes")
                #importances[output_type][model_name] = feature_importance(
                #    scaled_train[:,0:-1][neutral],
                #    scaled_train[:,-1][neutral],                    
                #    model_objects[model_name][output_type],
                #    mean_squared_error,
                #    x_columns=input_columns[output_type],
                #    col_start="neutral_")
                #
                # Compute feature importances for unstable regime 
                #
                #importances[output_type][model_name] = pd.concat([feature_importance(
                #    scaled_train[:,0:-1][unstable],
                #    scaled_train[:,-1][unstable],
                #    model_objects[model_name][output_type],
                #    mean_squared_error,
                #    x_columns=input_columns[output_type],
                #    col_start="unstable_"), importances[output_type][model_name]], axis=1)

                #
                # Compute feature importances for stable regime 
                #
                #importances[output_type][model_name] = pd.concat([feature_importance(
                #    scaled_train[:,0:-1][stable],
                #    scaled_train[:,-1][stable],
                #    model_objects[model_name][output_type],
                #    mean_squared_error,
                #    x_columns=input_columns[output_type],
                #    col_start="stable_"), importances[output_type][model_name]], axis=1)
                #
                # Output importance data to csv file
                #
                importances[output_type][model_name].to_csv(join(out_dir,
                                                                 output_type + "_" + model_name + "_importances.csv"),
                                                            index_label="input")
            
            
            # Unscale the Neural Networks predictions before calculating the metrics
            if model_name == "neural_network":
                scaled_test = pd.DataFrame()

                for model_metric in model_metric_types:
                    scaled_test.loc[:, (predictandLabel_model, model_metric)] = metrics[model_metric](data["test"][output_columns[output_type]].values, model_predictions[predictandLabel_model].values)
                    model_metrics.loc[predictandLabel_model, model_metric] = metrics[model_metric](data["test"][output_columns[output_type]].values, model_predictions[predictandLabel_model].values)
                    
                scaled_test.to_csv(join(out_dir, predictandLabel_model + "_scaled_metrics_NN.csv"), index=False)
                
                model_predictions[predictandLabel_model] = model_predictions[predictandLabel_model]  * input_scalers[output_type].scale_[len(input_scalers[output_type].scale_) - 1 ] + input_scalers[output_type].mean_[len(input_scalers[output_type].mean_) - 1 ]
                
            #
            # Compute error metrics
            #
            for model_metric in model_metric_types:
                model_metrics.loc[predictandLabel_model, model_metric] = metrics[model_metric](data["test"][output_columns[output_type]].values, model_predictions[predictandLabel_model].values)
                #
                # Output error metrics for user
                #
                print(f"{predictandLabel_model:30s} {model_metric:20s}: {model_metrics.loc[predictandLabel_model, model_metric]:.15f}")
 
            #
            # Save ML models
            #
            if model_name == "random_forest":
                #
                # Save trees as csv files
                #
                save_random_forest_csv(model_objects[model_name][output_type],
                                       np.array(input_columns[output_type]),
                                       out_dir, forest_name=output_type)
                #
                # Save as pickle file
                #
                pickle_filename = join(out_dir, f"{predictandLabel_model}.pkl")
                with open(pickle_filename, "wb") as pickle_file:
                    pickle.dump(model_objects[model_name][output_type], pickle_file)
                    
            elif model_name == "neural_network":
                #
                # save as *.h5 file
                #
                save_model(model_objects[model_name][output_type].model, join(out_dir, predictandLabel_model + ".h5"))
                
                #
                #
                #
                model_objects[model_name][output_type].save_fortran_model(join(out_dir, predictandLabel_model + "_fortran.nc"))
            #
            # Save ML models history (loss and training data)
            #
            if model_name == "neural_network":
                # Convert the history to a DataFrame and save it to a CSV file
                history_df = pd.DataFrame(history.history)
                history_df.to_csv(join(out_dir, predictandLabel_model + "_training_history.csv"), index=False)
                

        #
        # Saved the parameters for scaling the data for neural network as csv and pickle files
        #
        save_scaler_csv(input_scalers[output_type], var_scale_list,
                        join(out_dir, f"{output_type}_scale_values.csv"))
        scaler_filename = join(out_dir, f"{output_type}_scaler.pkl")
        with open(scaler_filename, "wb") as scaler_pickle:
            pickle.dump(input_scalers[output_type], scaler_pickle)

    #
    # Output the model metrics dataframe to csv
    #
    model_metrics.to_csv(join(out_dir, "surface_layer_model_metrics.csv"), index_label="Model")
    
    #
    # Output the model predictions dataframe to csv
    #
    model_predictions.to_csv(join(out_dir, "surface_layer_model_predictions.csv"), index_label="Time")
    return

if __name__ == "__main__":
    main()
