import numpy as np
import yaml
import argparse
from mlsurfacelayer.data import load_derived_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="Number of processes")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file)
    data_file = config["data_file"]
    train_test_split_date = config["train_test_split_date"]
    input_columns = config["input_columns"]
    output_columns = config["output_columns"]
    derived_columns = config["derived_columns"]

    return

if __name__ == "__main__":
    main()
