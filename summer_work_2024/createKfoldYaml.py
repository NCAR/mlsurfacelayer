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

    name = config['file_name'] # hold the current name temporarily
    print(name)
    
    out_dir = config['out_dir'] # hold the ouput directory name temporarily
    print(f'{out_dir}/{name}/')
    Path(f'{out_dir}/{name}/').mkdir(parents=True, exist_ok=True) # create sub-folder to hold the Kfold YAML's
    
    folds = config['k_fold_cross_validation']['N'] # retrieve number of Kfolds for this model
    print (folds)

    for i in range(folds):
        # rename vars
        config['k_fold_cross_validation']['k'] = i
        config['file_name'] = f'{name}--kfold-{str(i)}'
        # config['out_dir'] = f'{out_dir}{name}/{name}/model_QC_--kfold-{str(i)}'
        config['out_dir'] = f'{out_dir}/{name}/model_QC_--kfold-{str(i)}'

        # Save to a new YAML file
        #with open(out_dir + config['file_name'] + '.yml', 'w') as file:
        with open(f"{out_dir}/{name}/{config['file_name']}.yml", 'w') as file:
            yaml.dump(config, file)

if __name__ == '__main__':
    create_kfold_yamls()
