import numpy as np
import pandas as pd
import os
from os.path import exists, join


def save_scaler_csv(scaler_obj, input_columns, output_file):
    """
    Save the scaler information to csv so that it can be read later.

    Args:
        scaler_obj: Scikit-learn StandardScaler object
        input_columns:
        output_file:

    Returns:

    """
    input_scaler_df = pd.DataFrame({"mean": scaler_obj.mean_, "scale": scaler_obj.scale_},
                                   index=input_columns)
    input_scaler_df.to_csv(output_file, index_label="input")
    return input_scaler_df


def save_random_forest_csv(random_forest_model, features, out_path, forest_name="random_forest"):
    """
    Converts a scikit-learn random forest object into a set of csv files for each tree in the forest. If the
    specified directory does not currently exist, the function will create it.

    Args:
        random_forest_model: scikit learn RandomForestRegressor or RandomForestClassifier
        features: list or array of feature names in the order of training
        out_path: Path to directory containing random forest csv files.
        forest_name: Name of the forest model
    Returns:

    """
    if not exists(join(out_path, forest_name)):
        os.makedirs(join(out_path, forest_name))
    feature_frame = pd.DataFrame(features, columns=["feature"])
    feature_frame.to_csv(join(out_path, forest_name, f"{forest_name}_features.csv"), index_label="Index")
    rf_frames = random_forest_dataframes(random_forest_model, features)
    for r, rf_frame in enumerate(rf_frames):
        rf_frame.to_csv(join(out_path, forest_name, f"{forest_name}_tree_{r:04d}.csv"), float_format='%1.16e',
                        index_label="Node")
    return


def random_forest_dataframes(random_forest_model, feature_names):
    rf_frames = []
    for d, dt in enumerate(random_forest_model.estimators_):
        rf_frames.append(decision_tree_dataframe(dt, feature_names=feature_names))
    return rf_frames


def decision_tree_dataframe(decision_tree_model, feature_names=None):
    """
    Extracts the attributes of a decision tree into a DataFrame

    Args:
        decision_tree_model: scikit-learn DecisionTree object
        feature_names: array of names for each input feature in the order they were put into the tree
    Returns:
        :class:`pandas.DataFrame` : The decision tree represented as a table.
    """
    tree = decision_tree_model.tree_
    tree_dict = {}
    tree_vars = ["feature", "threshold", "value", "children_left", "children_right", "impurity"]
    if feature_names is not None:
        tree_vars.append("feature_name")
    for tree_var in tree_vars:
        if tree_var == "value":
            if tree.value.shape[2] > 1:
                # Assumes the tree value contains the number of instances in each class
                # Calculates the probability of the second class assuming the classes are 0 and 1
                tree_dict[tree_var] = tree.value[:, 0, 1] / tree.value[:, 0].sum(axis=1)
            else:
                tree_dict[tree_var] = tree.value[:, 0, 0]
        elif tree_var == "feature_name":
            tree_dict[tree_var] = feature_names[tree_dict["feature"]]
            tree_dict[tree_var][tree_dict["feature"] == -1] = "leaf node__"
        else:
            tree_dict[tree_var] = getattr(tree, tree_var)

    tree_frame = pd.DataFrame(tree_dict, columns=tree_vars)
    return tree_frame


def predict_decision_tree_frame(input_data, dt_frame):
    """
    Generate predictions for a single example from a decision tree.

    Args:
        input_data: 1D array of input data
        dt_frame: Decision tree in table format

    Returns:
        float: predicted value for input_data from tree.
    """
    index = 0
    not_leaf = True
    value = -9999.0
    while not_leaf:
        value = dt_frame.loc[index, "value"]
        if dt_frame.loc[index, "feature"] == -2:
            not_leaf = False
        else:
            exceeds = input_data[dt_frame.loc[index, "feature"]] > dt_frame.loc[index, "threshold"]
            if exceeds:
                index = dt_frame.loc[index, "children_right"]
            else:
                index = dt_frame.loc[index, "children_left"]
    return value
