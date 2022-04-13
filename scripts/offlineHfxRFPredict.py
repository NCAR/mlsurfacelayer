#!/usr/bin/env python

import pandas as pd
import numpy as np
from mlsurfacelayer.models import predict_decision_tree_frame
import os
filebase = "/Volumes/SuesRoo/mlsurfacelayer/FINO_WRF_2/modelsFrVelRF/friction_velocity40"
def main():
  wrfin  = pd.read_csv("/Volumes/SuesRoo/mlsurfacelayer/FINO_WRF_2/wrfCompare/ustar_inputs.csv")
  wrfin.index = wrfin["time"]
  wrfin = wrfin.drop(["time"], axis = 1)
  wrfout = pd.read_csv("/Volumes/SuesRoo/mlsurfacelayer/FINO_WRF_2/wrfCompare/outputs.csv")

  #myin = pd.read_csv("/Volumes/SuesRoo/mlsurfacelayer/FINO_WRF_2/modelsFrVelRF/surface_layer_model_predictions.csv")
  #wrfin = myin[["pressure:40_m:hPa","wave_dir_linear_interp:0_m:degrees","wave_height:0_m:m","wave_period:0_m:s","wind_speed:40_m:m_s-1","wind_direction:40_m:degrees","bulk_richardson:40_m:none","potential_temperature_gradient:20_m:K_m-1","wind_speed_gradient:20_m:s-1" ]]
  #wrfout = myin[["friction_velocity40-random_forest"]]

  files = os.listdir(filebase)

  offCount = 0
  for i in range(0,len(wrfout.index)):
     ave = 0
     count = 0
     for treeFile in files:
        if 'friction_velocity40_tree' in treeFile:
           filepath = os.path.join(filebase, treeFile)
           #print(filepath)
           trDf = pd.read_csv(filepath)
           #print(trDf.columns)
           #print( wrfin.iloc[i])
           #print wrfin.iloc[i]
           ave = ave + predict_decision_tree_frame(wrfin.iloc[i], trDf)
           count = count + 1
     ave = ave/count
     diff = abs(wrfout[" ustar"].iloc[i] - ave) 
     #diff = abs(wrfout[" hfx"].iloc[i] - ave)
     #diff = abs(wrfout["kin_heat_flux40-random_forest"].iloc[i] - ave)
     #diff = abs(wrfout["friction_velocity40-random_forest"].iloc[i] - ave)
 
     if diff > .0000001:
       #print(  i, " " , wrfout[" hfx"].iloc[i], " ", ave, " diff: ", wrfout[" hfx"].iloc[i] - ave  )
       #print(  i, " predOut: " , wrfout["kin_heat_flux40-random_forest"].iloc[i], " treeOut: ", ave  )
       print(  i, " " , wrfout[" ustar"].iloc[i], " ", ave,  " diff: ", wrfout[" ustar"].iloc[i] - ave  )
       #print(  i, " predOut: " , wrfout["friction_velocity40-random_forest"].iloc[i], " ", ave  )
       offCount+= 1
  print( "offCount ", offCount , " ", len(wrfout.index) )
if __name__ == "__main__":
    main()
