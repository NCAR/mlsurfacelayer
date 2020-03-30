************************
Machine Learining in WRF
************************
Once you have trained a ranndom forest or neural network
with Python machine learning models, one will want to
run them online within WRF.

Moving trained models to WRF
============================
#. Train machine learning models using the :code:`train_surface_models.py` script.
#. Copy the trained models to the directory where wrf.exe is located.

    #. Random Forest tree files should all be in the same directory with a tree_files.csv file containing the
       names of all the tree files.
    #. Neural network netCDF files should be together with the approporiate _scale_values.csv file.

#. Modify :code:`phys/module_physics_init.F` to contain the appropriate paths to the random forest
   and neural network files for each subset of the surface layer scheme.
#. Set the code in the namelist.input file to the ML surface layer scheme.
#. Compile WRF.
#. Run WPS and wrf.exe.

Adding a new machine learning parameterization to WRF
=====================================================
#. Add constants to :code:`Registry/Registry.EM_common` for each machine learning parameterization and ensure
   they do not clash with other constants for the same parameterization type.
#. Modify :code:`phys/module_physics_init.F` to call the init subroutine to load your machine learning models.
#. Modify :code:`phys/module_surface_driver.F` to call the machine learning surface layer parameterization subroutine.
#. Edit :code:`phys/Makefile` to add the new modules to the compilation list.
#. Compile WRF.