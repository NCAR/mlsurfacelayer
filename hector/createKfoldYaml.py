import yaml
import argparse
from pathlib import Path
import os

def create_kfold_yamls():
    # parser to recieve command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config YAML file")
    args = parser.parse_args()

    # open YAML w/ config info
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)

    prefix = '../config/' + name + '/'              # prefix to save all Kfold YAML's in same sub-folder
    Path(prefix).mkdir(parents=True, exist_ok=True) # create sub-folder to hold the Kfold YAML's

    out_dir = config['out_dir'] # hold the ouput directory name temporarily
    name = config['file_name'] # hold the current name temporarily
    folds = config['k_fold_cross_validation']['N'] # retrieve number of Kfolds for this model

    for i in range(folds):
        # rename vars
        config['k_fold_cross_validation']['k'] = i
        config['file_name'] = name + '--kfold-' + str(i)
        config['out_dir'] = out_dir + 'model_QC_' + '--kfold-' + str(i)

        # Save to a new YAML file
        with open(prefix + config['file_name'] + '.yml', 'w') as file:
            yaml.dump(config, file)

if __name__ == '__main__':
    create_kfold_yamls()