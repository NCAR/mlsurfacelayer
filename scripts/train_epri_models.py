#!/usr/bin/env python
import numpy as np
import yaml
import argparse
import pandas as pd
from mlsurfacelayer.data import load_derived_data
from mlsurfacelayer.epri_data import load_derived_data_random_test_train
from mlsurfacelayer.models import save_random_forest_csv, save_scaler_csv
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
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file)
    data_file = config["data_file"]
    out_dir = config["out_dir"]
    output_types = config["output_types"]
    input_columns = config["input_columns"]
    output_columns = config["output_columns"]
    output_col_names = [x for x in sorted(list(output_columns.values()))]
    derived_columns = config["derived_columns"]
    model_configs = config["model_config"]
    model_metric_types = config["model_metric_types"]
    data = load_derived_data_random_test_train(data_file)
    model_objects = dict()
    pred_columns = []
    for output_type in output_types:
        for model_name in model_configs.keys():
            pred_columns.append(output_type + "-" + model_name)
    model_predictions = pd.DataFrame(0, index=data["test"].index,
                                     columns=pred_columns + derived_columns + output_col_names,
                                     dtype=np.float32)
    model_predictions.loc[:, derived_columns] = data["test"][derived_columns]
    model_metrics = pd.DataFrame(0, index=pred_columns, columns=model_metric_types,
                                 dtype=np.float32)
    print(model_metrics)
    if not exists(out_dir):
        makedirs(out_dir)
    input_scalers = {}
    importances = {}
    for output_type in output_types:
        print(output_columns[output_type])
        print(data["train"][input_columns[output_type]].shape)
        print(data["train"][output_columns[output_type]].shape)
        print(data["train"][input_columns[output_type]].head())
        print(data["train"][output_columns[output_type]].head())

        print(data["test"][output_columns[output_type]].shape)
        model_predictions.loc[:, output_columns[output_type]] = data["test"][output_columns[output_type]]
        input_scalers[output_type] = StandardScaler()
        importances[output_type] = {}
        scaled_train_input = input_scalers[output_type].fit_transform(data["train"][input_columns[output_type]])
        scaled_test_input = input_scalers[output_type].transform(data["test"][input_columns[output_type]])
        for model_name, model_config in model_configs.items():
            model_objects[model_name] = {}
            print("Training", output_type, model_name)
            model_objects[model_name][output_type] = model_classes[model_name](**model_config)
            print(input_columns[output_type])
            print(data["train"][input_columns[output_type]].shape)
            if model_name == "random_forest":
                model_objects[model_name][output_type].fit(data["train"][input_columns[output_type]].values,
                                                           data["train"][output_columns[output_type]].values)
            else:
                model_objects[model_name][output_type].fit(scaled_train_input,
                                                           data["train"][output_columns[output_type]].values)
            print("Predicting", output_type, model_name)
            full_model = output_type + "-" + model_name
            if model_name == "random_forest":
                model_predictions.loc[:, full_model] = model_objects[
                    model_name][output_type].predict(data["test"][input_columns[output_type]])
                importances[output_type][model_name] = feature_importance(
                    data["train"][input_columns[output_type]].values,
                    data["train"][ output_columns[output_type]].values,
                    model_objects[model_name][output_type],
                    mean_squared_error,
                    x_columns=input_columns[output_type],
                    col_start="gen_")
                importances[output_type][model_name].to_csv(join(out_dir,
                                                                 output_type + "_" + model_name + "_importances.csv"),
                                                            index_label="input")
            else:
                model_predictions.loc[:, full_model] = model_objects[
                    model_name][output_type].predict(scaled_test_input)
                importances[output_type][model_name] = feature_importance(
                    scaled_train_input,
                    data["train"][output_columns[output_type]].values,
                    model_objects[model_name][output_type],
                    mean_squared_error,
                    x_columns=input_columns[output_type],
                    col_start="gen_")
                importances[output_type][model_name].to_csv(join(out_dir,
                                                                 output_type + "_" + model_name + "_importances.csv"),
                                                            index_label="input")
            for model_metric in model_metric_types:
                model_metrics.loc[full_model, model_metric] = metrics[
                    model_metric](data["test"][output_columns[output_type]].values,
                                  model_predictions[full_model].values)
                #print(f"{full_model:30s} {model_metric:20s}: {model_metrics.loc[full_model, model_metric]:0.5f}")
            if model_name == "random_forest":
                save_random_forest_csv(model_objects[model_name][output_type],
                                       np.array(input_columns[output_type]),
                                       out_dir, forest_name=output_type)
                pickle_filename = join(out_dir, f"{full_model}.pkl")
                with open(pickle_filename, "wb") as pickle_file:
                    pickle.dump(model_objects[model_name][output_type], pickle_file)
            elif model_name == "neural_network":
                save_model(model_objects[model_name][output_type].model, join(out_dir, full_model + ".h5"))
                model_objects[model_name][output_type].save_fortran_model(join(out_dir, full_model + "_fortran.nc"))

        save_scaler_csv(input_scalers[output_type], input_columns[output_type],
                        join(out_dir, f"{output_type}_scale_values.csv"))
        scaler_filename = join(out_dir, f"{output_type}_scaler.pkl")
        with open(scaler_filename, "wb") as scaler_pickle:
            pickle.dump(input_scalers[output_type], scaler_pickle)
    model_metrics.to_csv(join(out_dir, "epri_model_metrics.csv"), index_label="Model")
    model_predictions.to_csv(join(out_dir, "epri_predictions.csv"), index_label="DateTime")
    return





if __name__ == "__main__":
    main()
