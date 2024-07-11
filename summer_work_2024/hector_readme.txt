To create a Jupyterlab server:

1. Navigate to this url 'https://jupyterhub.hpc.ucar.edu/'
2. Click Production
3. Log in using your Username & Password
4. Open a Server
5. Now when you open a notebook, click on the top right where it says Kernel
6. Select the Kernel enviroment RAL (conda env RAL)
7. Now you are ready to run the cells inside notebook

Cells to run:

1. First cell with the imports
2. Second cell that is under the 'helper function' header
3. Skip section 'MVCO Raw file...'
4. In section 'MVCO file post-proccessed' run the first 2 cells
5. Skip the sections that are for graphing
6. Then you are at 'MOST computations' and can start running the cells below



'''make this a markdown file'''


# to create the data file with the derived variables run this line in terminal:

clone git url

cd mlsurfacelayer/scripts/

python process_mvco_data_qc.py -i "../summer_work_2024/data/eastData/qced_MVCO_ocn_sonic_vaisala_QC_all_data.2004-2023.csv" -o "../../data/mvco_mlsl_qc.csv"

python train_offshore_models_mvco.py ../../data/model_output/cv_test/cv_test--kfold-0.yml


there is a conda enviroment named 'ral' located in '/glade/campaign/ral/wsap/oracleMLSL/mlsurfacelayer/conda-envs/ral/'. add this to your conda enviroment file to easily acces it, typicaly this file should be located at '~/.conda/environments.txt' on casper.








in order to run the pbs script, you need to create 1 yaml file, this will act like the master file, in there you specify the name and the number of folds in the cross val. Then in the pbs script, you must change the array numbers from 0-9 to 0-2 or 0-4, to fit the number of kfolds you want, you also need to update the name var in the pbs script to match the name of the master file.






Hello! I am the author of this version of the repo, Hector.
This ReadMe will explain how to navigate/use the files to create and run a ML model.

1: Make sure you are connected to the proper tensorflow envirment.

2: Navigate to the folder called scripts. [command should be cd mlsurfacelayer/scripts]

3: Here you will execute "python process_mvco_data.py -i '' -o '' ". This will grab the preprocessed data file and process it agian to add in our derived variables.

4: Once you check the file was created properly, now you can run "python train_offshore_model.py config/trainblablalb.yaml"



To create a Jupyterlab server.

1. Navigate to this url 'https://jupyterhub.hpc.ucar.edu/'
2. Click Production
3. Log in using your Username & Password
4. Open a Server
5. Now when you open a notebook, click on the top right where it says Kernel
6. Select the Kernel enviroment RAL (conda env RAL)
7. Now you are ready to run the cells inside notebook
    (p.s. i think this wont work bc currently the enviroment is mine, so I have access to it since I 'know' the PATH, I will have to create a cell block that checks if you have access and if not grant access. or create the env in the folder where the guide will exist. Yea thats probably the better solution
5. Once inside on the left you can navigate to 
    glade/u/home/hmarrero/

location for mvco data analysis notebook (where MOST computation is done)
    glade/u/home/hmarrero/mlsurfacelayer/hector/mvco_data_analysis.ipynb

