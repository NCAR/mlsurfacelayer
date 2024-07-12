'''make this a markdown file'''

# First we have to clone the repo if we havnt already. 
#   I suggest creating a folder, for ex: outer, and cloning the repo there: outer/mlsurfacelayer.
# Here we are also seperating and cloning only the bracnh that we will need/use.
clone git --branch hector --single-branch "https://github.com/NCAR/mlsurfacelayer.git"

# Before we can create a model first we need to create the conda envirment neccesary to use this branch
#   This will create an envirment named 'ral' at this location 'outer/data/conda-envs/ral'
cd outer/mlsurfacelayer/summer_work_2024/conda_setup
sh setup_conda_RAL_env.sh

# Here is how to run the .sh file on your local machine that will train a model and draw the visuals automatically
    # First we go into the config folder to create the YAML file needed for the experiment.
    #   I suggest creating a copy of 'template.yml' and making your modifications to the experiment from there
    #   The only var that NEEDS to be changed for the new experiment is the file name, everything else is dependent your experiment goals
    cd outer/mlsurfacelayer/config
    mv template.yml custom_config.yml

    # For the purpose of this tutorial we will assume the name of the config you created has the name 'custom_config'
    # From within the config folder, navigate to the summer_work_2024 folder
    cd outer/mlsurfacelayer/summer_work_2024

    # Run the .sh script 'run.sh' and pass it the name 'custom_config'
    # This will automatically create the sub YAML's needed for cross validation, create and train the models, and lastly create the visuals.
    sh run.sh custom_config

        # To run the experiment on casper instead, you will need to perform the following
        # Modify the pbs script located at 'outer/mlsurfacelayer/summer_work_2024/job_main.pbs' ('job_tuner.pbs' if you want to perform hyperparameter tuning)
        # Change the name variable to what you named your YAML file, in this tutorial you would update it to name="custom_config"
        # Once the change is saved it is ready to run
        qsub job_main.pbs (or 'qsub job_tuner.pbs' if running hyperparameter tuning)

# Here are the individual steps used to train a model:
    # Run the following commands.

    # First we cd into that contains the python scripts and run the process data script.
    #   This will create the data file with the derived variables for the ML models to use.
    #   This script is also where the MOST is being computated
    cd outer/mlsurfacelayer/scripts/
    python process_mvco_data_qc.py -i "../summer_work_2024/data/eastData/qced_MVCO_ocn_sonic_vaisala_QC_all_data.2004-2023.csv" -o "../../data/mvco_mlsl_qc.csv"
    

    # Once the data file has been created succesfully we need to create a config file for your experiment
    #   I suggest creating a copy of 'template.yml' and making your modifications to the experiment from there
    #   The only var that NEEDS to be changed for the new experiment is the file name, everything else is dependent your experiment goals
    cd outer/mlsurfacelayer/config
    mv template.yml custom_config.yml

    # Now we run a script that will create for us the sub YAML's used for the cross validation scheme
    #   If you dont want to perform cross validtion you can skip this step
    cd outer/mlsurfacelayer/summer_work_2024
    python createKfoldYaml.py ../config/custom_config.yml

    # Once the config files have been created we can now run the script that trains the model.
    #   This will take as input the path to the YAML of the corresponding experiment
    #   Here you can pass in one of the cross validtion sub YAML's as shown below or the "master" YAML you manuely created
    #   Repeat this step for each sub YAML if performing cross validation
    cd outer/mlsurfacelayer/scripts
    python train_offshore_models_mvco.py ../../data/model_output/template/custom_config--kfold-0.yml

    # Finally once the model has been succesfully trained we need to draw our visuals
    #   Similar to the training script we will pass it the path to the config file
    cd outer/mlsurfacelayer/summer_work_2024
    python o_mvco_model_analysis.py ../../data/model_output/template/custom_config--kfold-0.yml

# Now you have everything you need to analyze the results of your experiment, good luck!