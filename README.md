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

To process raw data, run scripts/process_surface_data.py.

To train ML models, run scripts/train_surface_models.py and use the configuration files in the 
config folder to see how they work. 







