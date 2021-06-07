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
    derived_columns = config["derived_columns"]
    model_configs = config["model_config"]
    model_metric_types = config["model_metric_types"]
    stability_column = config["stability_column"]

    data = load_derived_data(data_file, train_test_split_date,
                             filter_counter_gradient=config["filter_counter_gradient"])

    if not exists(out_dir):
        makedirs(out_dir)

    input_scalers = {}
    importances = {}
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
        model_predictions.loc[date, "friction_velocity-mo"] = mo_out[0]
        model_predictions.loc[date, "temperature_scale-mo"] = mo_out[1]
        model_predictions.loc[date, "moisture_scale-mo"] = mo_out[2] * 1000.0

        for output_type in output_types:

            model_predictions.loc[:, output_columns[output_type]] = data["test"][output_columns[output_type]]
            input_scalers[output_type] = StandardScaler()
            importances[output_type] = {}
            scaled_train_input = input_scalers[output_type].fit_transform(data["train"][input_columns])
            scaled_test_input = input_scalers[output_type].transform(data["test"][input_columns])
            scaled_test_input_df = pd.DataFrame(scaled_test_input, columns=input_columns)

            full_model = output_type + "-" + "mo"
            for model_metric in model_metric_types:
                model_metrics.loc[full_model, model_metric] = metrics[model_metric](
                    data["test"][output_columns[output_type]].values, model_predictions[full_model].values)


    for model_name, model_config in model_configs.items():
        model_objects[model_name] = {}
        model_objects[model_name] = model_classes[model_name](**model_config)
        if model_name == "random_forest":
            model_objects[model_name].fit(data["train"][input_columns].values,
                                          data["train"][[*output_columns.values()]].values)
        else:
            model_objects[model_name].fit(scaled_train_input,
                                          data["train"][[*output_columns.values()]].values)

        for i, output_type in enumerate(output_types):
            print("Predicting", output_type, model_name)
            full_model = output_type + "-" + model_name
            unstable = data["train"][stability_column] < -0.02
            stable = data["train"][stability_column] > 0.02
            neutral = (data["train"][stability_column] >= -0.02) & (data["train"][stability_column] <= 0.02)
            if model_name == "random_forest":
                model_predictions.loc[:, full_model] = model_objects[
                                                           model_name].predict(data["test"][input_columns]).reshape(
                    len(data['test']), len(output_columns))[:, i]

            else:
                model_predictions.loc[:, full_model] = model_objects[
                                                           model_name].predict(scaled_test_input).reshape(
                    len(scaled_test_input), len(output_columns))[:, i]

            for model_metric in model_metric_types:
                model_metrics.loc[full_model, model_metric] = metrics[
                    model_metric](data["test"][output_columns[output_type]].values,
                                  model_predictions[full_model].values)


    model_metrics.to_csv(join(out_dir, "_metrics.csv"))
    model_objects[model_name].save_fortran_model(join(out_dir, "_fortran.nc"))
    save_scaler_csv(input_scalers[output_type], input_columns, join(out_dir, f"scale_values.csv"))


if __name__ == "__main__":
    main()
