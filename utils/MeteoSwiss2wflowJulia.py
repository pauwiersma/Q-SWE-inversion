#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:47:13 2023

@author: tesla-k20c
"""


import ewatercycle
import ewatercycle.parameter_sets
import ewatercycle.analysis
ewatercycle.CFG.load_from_file("/home/tesla-k20c/ssd/pau/ewatercycle/ewatercycle.yaml")
import logging
import warnings
import numpy as np
import pandas as pd
import glob
import os
from os.path import join

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("esmvalcore")
logger.setLevel(logging.WARNING)
import ewatercycle.forcing
import ewatercycle.models
import xarray as xr
import rioxarray as rioxr
from scipy import interpolate
import matplotlib.pyplot as plt
import time
ewatercycle.__version__

os.chdir("/home/tesla-k20c/ssd/pau/Github/ewc")
from SwissStations import *
#%%

# hydromt_path = "/mnt/scratch_pwiersma/PauWiersma/Data/HydroMT"
# staticmaps_hydroATLAS = xr.open_dataset(os.path.join(hydromt_path,"model_builds/Jonschwil_soilgrids2020_HydroATLAS/staticmaps.nc"))
# dem_withzeros = staticmaps_hydroATLAS.wflow_dem.fillna(0)

staticmaps = xr.open_dataset("/home/tesla-k20c/ssd/pau/ewatercycle/parameter-sets/wflow_julia_1000_thurJS/data/input/staticmaps_2017_plusRiverDepth.nc"
                                  , chunks = 10)
dem_withzeros = staticmaps.wflow_dem.fillna(0)

# staticmaps_30m = xr.open_dataset("/home/tesla-k20c/ssd/pau/ewatercycle/parameter-sets/wflow_julia_1000_thurJS/data/input/staticmaps_2017_plusRiverDepth.nc"
#                                   , chunks = 10)
# dem_withzeros = staticmaps_30m.wflow_dem.fillna(0)
### Different way: just take MS200m forcing and interpolate + mask
MS_forcing = xr.open_dataset("/home/pwiersma/scratch/Data/MeteoSwiss/wflow_MeteoSwiss_Thur_Jonschwil_1993_2017_test_pluspet.nc")
# MS_forcing = MS_forcing.isel({'time':slice(0,365)})
MS_forcing['pet'] = MS_forcing['pet'].fillna(MS_forcing['pet'].mean(dim = ['lat','lon']))
MS_forcing['tas'] = MS_forcing['tas'].fillna(MS_forcing['tas'].mean(dim = 'lon'))

# TTI = 2
# TT = 0 
# csnow = 1.2

# #partitionaccordig to Verseveld2022, 1 is 100% liquid, 0 is 100% solid
# partition = np.maximum(xr.zeros_like(MS_forcing['tas']),
#                              np.minimum(np.ones_like(MS_forcing['tas']),
#                                     (MS_forcing['tas']-TT-0.5*TTI)/TTI))
# snowfall_correction = np.absolute(1-partition) * (csnow-1) +1
# MS_forcing['pr'] *= snowfall_correction


MS_forcing = MS_forcing.interp_like(dem_withzeros,kwargs = dict(fill_value = 'extrapolate'))
MS_forcing.coords['mask'] = xr.where(staticmaps.wflow_subcatch==1,1,0)

print('wflow_dem has ', np.count_nonzero(np.isnan(staticmaps.wflow_dem.data)).compute(),' nans')
print('MS_Frocing has ', np.count_nonzero(np.isnan(MS_forcing.pr[0].data)),' nans')


for var in ['pr','tas','pet']:
    MS_forcing[var] = xr.where(dem_withzeros ==0, np.nan, MS_forcing[var])
# test = MS_forcing[['time','y','x','precip','temp','pet']]
MS_forcing  = MS_forcing.drop_dims('bnds').rename({'lat':'y','lon':'x','pr':'precip','tas':'temp'})
MS_forcing['time'] = MS_forcing.time - np.timedelta64(12,'h')
MS_forcing['time'] = MS_forcing.time - np.timedelta64(30,'m')
MS_forcing = MS_forcing.transpose('time','y','x')
MS_forcing.coords['height'] = 2.0
MS_forcing['height'].attrs['unit'] = 'm'
MS_forcing['height'].attrs['standard_name'] = 'height'
MS_forcing['height'].attrs['long_name'] = 'height'
MS_forcing['height'].attrs['positive'] = 'up'

# diff = xr.where((~np.isnan(staticmaps.wflow_dem.data) & np.isnan(MS_forcing.precip[0].data)),1e6,staticmaps.wflow_dem)
# #count nans
# print('wflow_dem has ', np.count_nonzero(np.isnan(staticmaps.wflow_dem.data)).compute(),' nans')
# print('MS_Frocing has ', np.count_nonzero(np.isnan(MS_forcing.precip[0].data)).compute(),' nans')
# #So MS_forcing has more nanas
# #TODO fill the zero in the next line proberply
# # MS_forcing_correct = xr.where(~(np.isnan(staticmaps.wflow_dem.data) & np.isnan(MS_forcing.precip[0].data)),0,MS_forcing)
# print('MS_Frocing has ', np.count_nonzero(np.isnan(MS_forcing_correct.precip[0].data)).compute(),' nans')


forcing_dir = "/home/tesla-k20c/ssd/pau/ewatercycle/parameter-sets/wflow_julia_1000_thurJS/data/input/forcing_jonschwil_MS_1000_1993_2017.nc"
MS_forcing.to_netcdf(forcing_dir)
      