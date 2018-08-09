import pandas as pd
from glob import glob
from os.path import join


def process_idaho_data(csv_path):
    """
    This function loads all of the cabauw data files and then calculates the relevant derived quantities necessary
    to build the machine learning parameterization.

    Args:
        csv_path: Path to all csv files.

    Returns:

    """
    csv_files = sorted(glob(join(csv_path, "*.csv")))
    file_types = [csv_file.split(".")[0].split("_")[1] for csv_file in csv_files]
    data = dict()
    for c, csv_file in enumerate(csv_files):
        data[file_types[c]] = pd.read_csv(csv_file, na_values=[-9999.0])
        data[file_types[c]].index = pd.to_datetime(data[file_types[c]]["TimeStr"], format="%Y%m%d.%H:%M")
    combined_data = pd.concat(data, axis=1, join="inner")
    return combined_data

