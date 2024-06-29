import yaml
import argparse
from pathlib import Path
import os

def create_kfold_yamls():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config yaml file")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)

    name = config['file_name']
    prefix = '../config/' + name + '/' 
    out_dir = config['out_dir']
    Path(prefix).mkdir(parents=True, exist_ok=True)  

    folds = config['k_fold_cross_validation']['N']

    for i in range(folds):
        config['k_fold_cross_validation']['k'] = i
        config['file_name'] = name + '--kfold-' + str(i)
        config['out_dir'] = out_dir + 'model_QC_' + '--kfold-' + str(i)

        # Save to a YAML file
        with open(prefix + config['file_name'] + '.yml', 'w') as file:
            yaml.dump(config, file)

   
