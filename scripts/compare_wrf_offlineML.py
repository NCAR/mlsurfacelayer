#!/usr/bin/env python

import pandas as pd
import numpy as np


def main():
  offlineOut = pd.read_csv( "/Volumes/SuesRoo/mlsurfacelayer/FINO_WRF_RF_PATS_TEST/models/surface_layer_model_predictions.csv")
  offlineOut["Time"] = pd.to_datetime(offlineOut["Time"])
  offlineOut.index = offlineOut["Time"]
  offlineOut = offlineOut.drop(columns = ["Time"])

  wrfout = pd.read_csv("~/Downloads/outputs.csv")
  # 'IN01', ' IN02', ' IN03', ' IN04', ' IN05', ' IN06', ' IN07', ' IN08',
  inputList = ['global_horizontal_irradiance:30_m:W_m-2', 'angleBetweenWaveWind',
     'temperature:40_m:K', 'water_sfc_temperature:0_m:K',
     'pressure:40_m:hPa', 'potential_temperature:40_m:K',
     'skin_virtual_potential_temperature:0_m:K', 'mixing_ratio:40_m:g_kg-1',
     'relative_humidity:40_m:%', 'wave_dir_linear_interp:0_m:degrees',
     'wave_height:0_m:m', 'wave_period:0_m:s', 'wave_phase_speed:0_m:m_s-1',
     'wind_speed:40_m:m_s-1', 'wind_direction:40_m:degrees',
     'bulk_richardson:40_m:none',
     'potential_temperature_gradient:20_m:K_m-1',
     'wind_speed_gradient:20_m:s-1']
     #' IN09', ' IN10', ' IN11', ' IN12', ' IN13', ' IN14', ' IN15', ' IN16',
     #' IN17', ' IN18', ' ustar', ' tstar', ' qstar'
  #wrfout = wrfout.rename(columns={ 'IN01':'global_horizontal_irradiance:30_m:W_m-2' , 
  #                        ' IN02':'angleBetweenWaveWind' ,
  #                        ' IN03':'temperature:40_m:K' ,
  #                        ' IN04':'water_sfc_temperature:0_m:K' ,
  #                        ' IN05':'pressure:40_m:hPa' ,
  #                        ' IN06':'potential_temperature:40_m:K' ,
  #                        ' IN07':'skin_virtual_potential_temperature:0_m:K' ,
  #                        ' IN08': 'mixing_ratio:40_m:g_kg-1' ,
  #                        ' IN09':'relative_humidity:40_m:%' ,
  #                        ' IN10':'wave_dir_linear_interp:0_m:degrees' ,
  #                        ' IN11':'wave_height:0_m:m' ,
  #                        ' IN12':'wave_period:0_m:s' ,
  #                        ' IN13':'wave_phase_speed:0_m:m_s-1' ,
  #                        ' IN14':'wind_speed:40_m:m_s-1' ,
  #                        ' IN15':'wind_direction:40_m:degrees',
  #                        ' IN16':'bulk_richardson:40_m:none',
  #                        ' IN17':'potential_temperature_gradient:20_m:K_m-1' ,
  #                        ' IN18':'wind_speed_gradient:20_m:s-1'})
  #wrfout = wrfout[ ['IN01', ' ustar', ' tstar', ' qstar']]
  
  for i in range (0,len(wrfout.index)):
     for j in range(0,len(inputList)):
        if abs( wrfout[wrfout.columns[j]].iloc[i] - offlineOut[inputList[j]].iloc[i]) > .00005:
           print(  inputList[j] , " ", wrfout[wrfout.columns[j]].iloc[i], " ",  offlineOut[inputList[j]].iloc[i])  
     #print( wrfout[' ustar'].iloc[i], " ",  offlineOut['friction_velocity40-random_forest'].iloc[i], " ", wrfout['IN01'].iloc[i], " ", offlineOut['global_horizontal_irradiance:30_m:W_m-2'].iloc[i]) 
     #print( abs(wrfout[' ustar'].iloc[i] - offlineOut['friction_velocity40-random_forest'].iloc[i])/offlineOut['friction_velocity40-random_forest'].iloc[i])

if __name__ == "__main__":
    main()
