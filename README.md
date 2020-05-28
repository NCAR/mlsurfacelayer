# mlsurfacelayer
Machine learning surface layer parameterizations.

## Requirements
The mlsurfacelayer library requires Python 3.6 or later 
and the following python libraries:
* numpy
* matplotlib
* scipy
* pandas
* xarray
* tensorflow
* scikit-learn
* pyyaml
* netcdf4
* numba


## Installation
Install the miniconda python distribution in your chosen directory.
The $ indicates command line inputs and should not be copied into your terminal. 
```bash
$ wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ sh Miniconda3-latest-Linux-x86_64.sh -b
```
Include the base miniconda bin directory in your `$PATH` environment variable. Change `$HOME` to the
appropriate path if you are not installing miniconda in your home directory.
```bash
$ export PATH="$HOME/miniconda/bin:$PATH"
```
Add the conda-forge channel and set it as the priority channel to prevent conflicts between the Anaconda
and conda-forge versions of core libraries like numpy.
```bash
$ conda config --add channels conda-forge
$ conda config --set channel_priority strict
```
Now, create a new conda environment with the main dependencies (except tensorflow) installed in it.
The --yes command will automatically install everything without any extra confirmation from you.
```bash
$ conda create -n sfc --yes -c conda-forge python=3.7 pip numpy scipy matplotlib pandas xarray pyyaml netcdf4 scikit-learn tqdm pytest
```
To activate the newly created environment, run the following command.
```bash
$ source activate sfc
```
You can also edit your $PATH environment variable directly. Note that the environment path may
differ depending on the version of miniconda you are using.
```bash
$ export PATH="$HOME/miniconda/envs/sfc/bin:$PATH"
```
Verify that the correct Python environment is being used.
```bash
$ which python
```
Install tensorflow using pip. There is a tensorflow channel on conda-forge, but I have experienced issues 
with it in the past. For this package, I recommend using tensorflow 2.1.0, but any version of tensorflow
beyond 2.0.0 should work (barring future significant API changes).
```bash
$ pip install tensorflow==2.1.0
```

To install the mlsurfacelayer library in your python environment, go to the
top level directory and run 

`$ pip install .`

To import modules after installing in a script, use the following type of command:

`from mlsurfacelayer.derived import air_density`

## How to Run
First download surface layer data to your local machine.
* [Idaho Raw Data](https://drive.google.com/open?id=1BLHVgtWdY_H7230QwabeUv0xNJX59xyd)
* [Cabauw Raw Data](https://drive.google.com/open?id=10x4VeF3yJmyWv5LV8kkNJ5vbkxME4fJc)
* [Cabauw Processed Data](https://drive.google.com/open?id=1AXdpqMcRmQsbzSASyD1yb7wLqT1rBQxQ)

### scripts/process_surface_data.py
To process raw data, run scripts/process_surface_data.py. The script is configured by changing the command
line arguments:
* -i, --input: Path to the directory containing the raw CSV files for Idaho or Cabauw
* -o, --output: Filename and path for the csv file containing all of the derived data. It will be created
by the script, and the older version of the file will be overwritten.
* -s, --site: Currently "idaho" and "cabauw" are the valid options. Additional sites can be added
under other if clauses.
* -w, --wind: Size of the time averaging window in pandas string time units (i.e., "30Min"). The data are 10-minute averaged by default.
30 minute averages seem to peform better in initial tests of the system.
* -r, --refl: If used, changes the sign of fluxes that are counter the temperature or moisture gradient. 
Only implemented for Cabauw and not a recommended option going forward.

The functions for each site are in data.py. The functions for calculating the derived quantities are in derived.py. 

### scripts/train_surface_models.py
To train ML models, run scripts/train_surface_models.py and use the configuration files in the 
config folder to see how they work.  The file `config/surface_layer_training_30Min_20191121.yml` is the most
up-to-date version. Currently only Cabauw models have been trained with this, but it should work for Idaho and 
other sites that are included in process_surface_data.py.

To run `train_surface_models.py`:
* Make sure the latest version of the code is installed to your python environment.
* `cd scripts`
* `python train_surface_models.py ../config/surface_layer_training_30Min_20191121.yml`

The config file contains the following arguments:
* data_file: Path to the derived data file from process_surface_data.py
* out_dir: Path to the directory where trained models, metrics and interpretation information are output.
* filter_counter_gradient: If 1 change the sign of fluxes that are opposite the temperature gradient. If 0, do not. 
* train_test_split_date: Date in "YYYY-MM-DD" format that separates training and testing data periods. 
Examples before the split data are used for training, and examples afterward are used for testing.
* output_types: Names for the different ML models. Should be `["friction_velocity", "temperature_scale", "moisture_scale"]`
* model_metric_types: List of metrics calculated for each output type. Currently we use
`["mean_squared_error", "mean_absolute_error", "mean_error", "pearson_r2", "hellinger_distance"]`
* input_columns: For each output type, a list of the input variable names in `data_file`.
* output_columns: The columns in `data_file` associated with each output type
* derived_columns: A list of additional data columns to include with the predictions for analysis purposes.
stability_column: The column to use when splitting the data into stability regimes.
* model_config: Specify the hyperparameters for different machine learning models. Currently `random_forest` and `neural_network` 
    are the supported types. If a parameter is not specified, the default value is used.
  * random_forest: scikit-learn random forest regressor
    * n_estimators: Number of trees
    * max_features: Number of features randomly selected at each node for evaluation
    * n_jobs: Number of processes used for training
    * max_leaf_nodes: Max number of leaf nodes for each tree. Used to control depth
  * neural_network: Dense (fully-connected) neural network implemented in Tensorflow
    * hidden_layers: Number of hidden layers
    * hidden_neurons: Number of neurons per hidden layer
    * batch_size: Number of examples per training batch (random sample from full training set)
    * epochs: Number of iterations over training set
    * lr: Learning rate.
    * activation: Non-linear activation function. Options include "relu", "tanh", "elu", "selu"
    * l2_weight: Strength of the l2 norm (Ridge) regularization penalty. Can help with overfitting







