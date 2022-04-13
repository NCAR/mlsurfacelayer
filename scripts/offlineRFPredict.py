#!/usr/bin/env python

import pandas as pd
import numpy as np
from mlsurfacelayer.models import predict_decision_tree_frame
import os
filebase = "/Volumes/SuesRoo/mlsurfacelayer/FINO_WRF_2/modelsKinSensHeatFluxRF_potTwaterSfc/kin_heat_flux40"
def main():
  wrfin  = pd.read_csv("/Volumes/SuesRoo/mlsurfacelayer/FINO_WRF_2/wrfCompare/hfx_inputs.csv")
  wrfin.index = wrfin["time"]
  wrfin = wrfin.drop(["time"], axis = 1)
  wrfout = pd.read_csv("/Volumes/SuesRoo/mlsurfacelayer/FINO_WRF_2/wrfCompare/outputs.csv")
  files = os.listdir(filebase)

  offCount = 0
  for i in range(0,len(wrfout.index)):
     ave = 0
     count = 0
     for treeFile in files:
        if 'kin_heat_flux40_tree' in treeFile:
           filepath = os.path.join(filebase, treeFile)
           #print(filepath)
           trDf = pd.read_csv(filepath)
           #print(trDf.columns)
           #print( wrfin.iloc[i])
           #print wrfin.iloc[i]
           ave = ave + predict_decision_tree_frame(wrfin.iloc[i], trDf)
           count = count + 1
     ave = ave/count
     diff = abs(wrfout[" hfx"].iloc[i] - ave)
     if diff > .0000001:
       print(  i, " " , wrfout[" hfx"].iloc[i], " ", ave  )
       offCount+= 1
  print( "offCount ", offCount , " ", len(wrfout.index) )
if __name__ == "__main__":
    main()
