import numpy as np
import pandas as pd
import os
from os.path import exists, join
from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
import xarray as xr


class DenseNeuralNetwork(object):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.

    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        inputs: Number of input values
        outputs: Number of output values
        activation: Type of activation function
        output_activation: Activation function applied to the output layer
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object
        use_noise: Whether or not additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        use_dropout: Whether or not Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        verbose: Level of detail to provide during training
        model: Keras Model object
    """
    def __init__(self, hidden_layers=1, hidden_neurons=4, activation="relu",
                 output_activation="linear", optimizer="adam", loss="mse", use_noise=False, noise_sd=0.01,
                 lr=0.001, use_dropout=False, dropout_alpha=0.1, batch_size=128, epochs=2,
                 l2_weight=0.01, sgd_momentum=0.9, adam_beta_1=0.9, adam_beta_2=0.999, decay=0, verbose=0,
                 classifier=False):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.loss = loss
        self.lr = lr
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.decay = decay
        self.verbose = verbose
        self.classifier = classifier
        self.y_labels = None
        self.model = None
        self.optimizer_obj = None

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        nn_input = Input(shape=(inputs,), name="input")
        nn_model = nn_input
        for h in range(self.hidden_layers):
            nn_model = Dense(self.hidden_neurons, activation=self.activation,
                             kernel_regularizer=l2(self.l2_weight), name=f"dense_{h:02d}")(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(nn_model)
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(nn_model)
        nn_model = Dense(outputs,
                         activation=self.output_activation, name=f"dense_{self.hidden_layers:02d}")(nn_model)
        self.model = Model(nn_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2, decay=self.decay)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum, decay=self.decay)
        self.model.compile(optimizer=self.optimizer_obj, loss=self.loss)

    def fit(self, x, y):
        inputs = x.shape[1]
        if len(y.shape) == 1:
            outputs = 1
        else:
            outputs = y.shape[1]
        if self.classifier:
            outputs = np.unique(y).size
        self.build_neural_network(inputs, outputs)
        if self.classifier:
            self.y_labels = np.unique(y)
            y_class = np.zeros((y.shape[0], self.y_labels.size), dtype=np.int32)
            for l, label in enumerate(self.y_labels):
                y_class[y == label, l] = 1
            self.model.fit(x, y_class, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
        else:
            self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
        return

    def save_fortran_model(self, filename):
        """
        Save neural network weights to a netCDF file that can be read by module_neural_net.f90.

        Args:
            filename: Name of the neural network file.

        Returns:

        """
        nn_ds = xr.Dataset()
        num_dense = 0
        layer_names = []
        for layer in self.model.layers:
            if "dense" in layer.name:
                layer_names.append(layer.name)
                dense_weights = layer.get_weights()
                nn_ds[layer.name + "_weights"] = ((layer.name + "_in", layer.name + "_out"), dense_weights[0])
                nn_ds[layer.name + "_bias"] = ((layer.name + "_out",), dense_weights[1])
                nn_ds[layer.name + "_weights"].attrs["name"] = layer.name
                nn_ds[layer.name + "_weights"].attrs["activation"] = layer.get_config()["activation"]
                num_dense += 1
        nn_ds["layer_names"] = (("num_layers",), np.array(layer_names))
        nn_ds.attrs["num_layers"] = num_dense
        nn_ds.to_netcdf(filename, encoding={'layer_names':{'dtype': 'S1'}})
        return

    def predict(self, x):
        if self.classifier:
            y_prob = self.model.predict(x, batch_size=self.batch_size)
            y_out = self.y_labels[np.argmax(y_prob, axis=1)].ravel()
        else:
            y_out = self.model.predict(x, batch_size=self.batch_size).ravel()
        return y_out

    def predict_proba(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        return y_prob


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
    with open(join(out_path, forest_name, "tree_files.csv"), "w") as tree_file:
        for r, rf_frame in enumerate(rf_frames):
            tree_name = f"{forest_name}_tree_{r:04d}.csv"
            rf_frame.to_csv(join(out_path, forest_name, tree_name), float_format='%1.16e',
                            index_label="Node")
            tree_file.write(tree_name + "\n")
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
