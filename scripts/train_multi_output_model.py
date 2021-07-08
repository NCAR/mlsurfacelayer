import pandas as pd
from mlsurfacelayer.data import load_derived_data
import yaml
import argparse
from mlsurfacelayer.mo import mo_similarity
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlsurfacelayer.metrics import mean_error, hellinger_distance, pearson_r2
from sklearn.preprocessing import StandardScaler
from mlsurfacelayer.models import DenseNeuralNetwork, save_scaler_csv
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import load_model
from os.path import join, exists
from matplotlib.colors import LogNorm
import joblib
from os import makedirs
import matplotlib.pyplot as plt
from tensorflow.keras.models import save_model

metrics = {"mean_squared_error": mean_squared_error,
           "mean_absolute_error": mean_absolute_error,
           "pearson_r2": pearson_r2,
           "hellinger_distance": hellinger_distance,
           "mean_error": mean_error}

model_classes = {"random_forest": RandomForestRegressor,
                 "neural_network": DenseNeuralNetwork}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    data_file = config["data_file"]
    train_test_split_date = pd.Timestamp(config["train_test_split_date"])
    out_dir = config["out_dir"]
    output_types = config["output_types"]
    input_columns = config["input_columns"]
    output_columns = config["output_columns"]
    output_column_list = [output_columns[output_type] for output_type in output_types]
    derived_columns = config["derived_columns"]
    model_configs = config["model_config"]
    model_metric_types = config["model_metric_types"]
    stability_column = config["stability_column"]

    data = load_derived_data(data_file, train_test_split_date,
                             filter_counter_gradient=config["filter_counter_gradient"])

    if not exists(out_dir):
        makedirs(out_dir)

    model_objects = {}
    pred_columns = []

    for output_type in output_types:
        for model_name in model_configs.keys():
            pred_columns.append(output_type + "-" + model_name)
        pred_columns.append(output_type + "-" + "mo")
    model_predictions = pd.DataFrame(0, index=data["test"].index,
                                     columns=pred_columns + derived_columns + [*output_columns.values()],
                                     dtype=np.float32)
    model_predictions.loc[:, derived_columns] = data["test"][derived_columns]
    model_metrics = pd.DataFrame(0, index=pred_columns, columns=model_metric_types,
                                 dtype=np.float32)

    print("Calculating Monin Obukhov...")
    for d, date in enumerate(data["test"].index):
        mo_out = mo_similarity(data["test"].loc[date, "u_wind:10_m:m_s-1"],
                               data["test"].loc[date, "v_wind:10_m:m_s-1"],
                               data["test"].loc[date, "skin_temperature:0_m:K"],
                               data["test"].loc[date, "temperature:10_m:K"],
                               data["test"].loc[date, "skin_saturation_mixing_ratio:0_m:g_kg-1"] / 1000.0,
                               data["test"].loc[date, "mixing_ratio:2_m:g_kg-1"] / 1000.0,
                               data["test"].loc[date, "pressure:2_m:hPa"],
                               mavail=data["test"].loc[date, "moisture_availability:soil:None"],
                               z0=0.017,
                               z10=10.0,
                               z2=10.0)
        for i, output_type in enumerate(output_types):
            mo_string = output_type + '-mo'
            if output_type != 'moisture_scale':
                model_predictions.loc[date, mo_string] = mo_out[i]
            else:
                model_predictions.loc[date, mo_string] = mo_out[i] * 1000

    input_scaler, output_scaler = StandardScaler(), StandardScaler()
    scaled_train_input = input_scaler.fit_transform(data["train"][input_columns])
    scaled_train_output = output_scaler.fit_transform(data["train"][output_column_list])
    scaled_test_input = input_scaler.transform(data["test"][input_columns])

    scaled_test_input_df = pd.DataFrame(scaled_test_input, columns=input_columns)

    for output_type in output_types:

        model_predictions.loc[:, output_columns[output_type]] = data["test"][output_columns[output_type]]
        full_model = output_type + "-" + "mo"
        for model_metric in model_metric_types:
            model_metrics.loc[full_model, model_metric] = metrics[model_metric](
                data["test"][output_columns[output_type]].values, model_predictions[full_model].values)

    for model_name, model_config in model_configs.items():
        model_objects[model_name] = {}
        model_objects[model_name] = model_classes[model_name](**model_config)
        if model_name == "random_forest":
            model_objects[model_name].fit(data["train"][input_columns],
                                          data["train"][output_column_list])

            preds = model_objects[model_name].predict(data["test"][input_columns]).reshape(
                                            len(data['test']), len(output_column_list))
        elif model_name == 'neural_network':
            model_objects[model_name].fit(scaled_train_input, scaled_train_output)

            preds = output_scaler.inverse_transform(model_objects[model_name].predict(scaled_test_input).reshape(
                    len(scaled_test_input), len(output_column_list)))

        for i, output_type in enumerate(output_types):
            print("Predicting", output_type, model_name)
            full_model = output_type + "-" + model_name

            model_predictions.loc[:, full_model] = preds[:, i]

            for model_metric in model_metric_types:
                model_metrics.loc[full_model, model_metric] = metrics[
                    model_metric](data["test"][output_columns[output_type]].values,
                                  model_predictions[full_model].values)

    model_objects['neural_network'].save_fortran_model(join(out_dir, "NN_fortran.nc"))
    save_model(model_objects['neural_network'].model, join(out_dir, "NN_tensorflow.h5"))
    model_metrics.to_csv(join(out_dir, "NN_metrics.csv"))
    model_predictions.to_csv(join(out_dir, "NN_predictions.csv"))
    save_scaler_csv(input_scaler, input_columns, join(out_dir, f"input_scale_values.csv"))
    save_scaler_csv(output_scaler, output_column_list, join(out_dir, f"output_scale_values.csv"))


if __name__ == "__main__":
    main()
