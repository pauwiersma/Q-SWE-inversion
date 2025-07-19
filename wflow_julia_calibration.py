#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:49:49 2023

@author: tesla-k20c
"""
#%%
import logging
import warnings
import numpy as np
import pandas as pd
import glob
import os
from os.path import join
import rasterio as rio
# from pathos.threading import ThreadPool as Pool
# import tomli
# import tomli_w
import tomlkit
import shutil

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("esmvalcore")
logger.setLevel(logging.WARNING)
# import ewatercycle.forcing
# import ewatercycle.models
import xarray as xr
import rioxarray as rioxr
from scipy import interpolate
import matplotlib.pyplot as plt 
import time
from datetime import date
from datetime import datetime
import HydroErr as he
# ewatercycle.__version__

EWC_ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')
EWC_RUNDIR = os.getenv('EWC_RUNDIR', default = '/home/pwiersma/scratch/Scripts/Github/ewc/')
os.chdir(EWC_RUNDIR)
# from SwissStations import *
from SnowClass import *
from RunWflow_Julia import *


from pathlib import Path
import ewatercycle
import numpy as np
from IPython.display import clear_output

# os.chdir("/home/pwiersma/")

from ewatercycle_wflowjl.forcing.forcing import WflowJlForcing
from ewatercycle_wflowjl.model import WflowJl
from ewatercycle_wflowjl.utils import get_geojson_locs

# os.chdir("/home/pwiersma/scratch/Scripts/Github/ewc")

# %matplotlib ipympl
# %matplotlib inline


#%%


RUN_BASIS = 'mwfets'
resolution = '1000m'
START_YEAR = 1998 #from 1993 for ERA5 variables for now
END_YEAR  =2022
# END_YEAR  =2002

par = [1]
# par = [0.01,0.1,0.4,0.7,1.0]
# par = [0.01,1]
# par = [0.0,0.001]#,1.2]
# par = [True,False]
# par = [-1,0,1]
# par = [1,25,50,100]
# par = [1,3,5]
# par = [2]
# par = [100,200,300,400,500]
# par = [0.5,1,1.5,4,8]
# par = [None, {'KsatHorFrac': '/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV/synthetic_obs/Errors_2009_2015/KsatHorFrac_synthetic_Landwasser_Errors_2009_2015_soilscale_3.txt',
#  'KsatVer': '/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV/synthetic_obs/Errors_2009_2015/KsatVer_synthetic_Landwasser_Errors_2009_2015_soilscale_3.txt',
#  'f': '/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV/synthetic_obs/Errors_2009_2015/f_synthetic_Landwasser_Errors_2009_2015_soilscale_3.txt'}]
# par = ['Default',
#         'SoilThicknessvalley0.3',
#         'SoilThicknessriver4',
#         'SoilThicknessriver0.2']
# par = ['/home/pwiersma/scratch/Data/ewatercycle/experiments/data/input/wflow_MeteoSwiss_1000m_Dischma_2014_2020_rain.nc',
#       '/home/pwiersma/scratch/Data/ewatercycle/experiments/data/input/wflow_MeteoSwiss_1000m_Dischma_2014_2020_snow.nc']
# par = np.arange(0,1.1,0.2)
# par = [5,15]
parname = "mwf"
# parname = "path_forcing"

# basins = ['Jonschwil','Landquart','Landwasser','Rom','Verzasca']
# basins=  ['Mogelsberg','Dischma','Ova_da_Cluozza','Sitter','Werthenstein',
#               'Sense','Ilfis','Eggiwil','Chamuerabach','Veveyse','Minster','Ova_dal_Fuorn','Alp','Biber','Muota','Riale_di_Calneggia',
#               'Chli_Schliere','Allenbach']#'Kleine_Emme','Riale_di_Pincascia',
basins = ['Dischma']
time0 = time.time()
for basin in basins:
    RUN_NAME = f"{basin}_{RUN_BASIS}"
      

    # parameter_set = summer_calib
    # parameter_set['DD_max'] = 5
    # parameter_set['DD_min'] = 2
    # parameter_set['sfcf'] = 1.2
    # parameter_set['sfcf_scale'] = 1.5
    # parameter_set['silent'] = False
    # parameter_set['TT'] = 1
    
    
    model_dics = {}
    
    for ii, k in enumerate(par):
        # parameter_set = posterior_parset[np.random.randint(0,len(posterior_parset))]
        # parameter_set = posterior_parset[0]
        # del parameter_set
        parameter_set = {}
        parameter_set['KsatHorFrac'] = 100
        parameter_set['f'] = 1
        parameter_set['c'] = 1
        parameter_set['KsatVer'] = 1
        # parameter_set['TT'] = -0
        # parameter_set['SoilThickness'] = 1
        # if k ==True and k!= 1:
        #     parameter_set['sfcf'] = 0        
        #     parameter_set['masswasting'] = False
        #     parameter_set['DD'] ='static'
        # else:
        #     parameter_set['sfcf'] = 1.2
        #     parameter_set['sfcf_scale'] = 1.2
        #     parameter_set['masswasting'] = True
        #     parameter_set['mwf'] = 0.5        
        #     parameter_set['DD'] ='Hock'
# trueparams.loc['sfcf'] = 1.2
#             trueparams.loc['sfcf_scale'] = 1.2
#             trueparams.loc['TT'] = 0.5
#             trueparams.loc['tt_scale'] = -0.3
#             trueparams.loc['rfcf'] = 1.2
        parameter_set['rfcf'] = 1
        # parameter_set['thetaR'] = 1
        parameter_set['thetaS'] = 1
        parameter_set['InfiltCapSoil'] =1
        # parameter_set['Cfmax'] = 20
        parameter_set['petcf_seasonal'] = True
        # parameter_set['TI'] = True
        parameter_set['DD_min'] = 2
        parameter_set['DD_max'] = 5
        # parameter_set['N'] = 1
        parameter_set['CV'] = 0.272731
        # parameter_set[]

        parameter_set['DD'] = 'Hock'
        parameter_set['m_hock'] = 3.291541
        parameter_set['r_hock'] = 0.008541
        parameter_set['vegcf'] = 0
        parameter_set['WHC'] = 0.2

        parameter_set['NC_STATES'] = None
        parameter_set['TT'] = 0.566731
        parameter_set['TTM'] = parameter_set['TT'] 
        parameter_set['tt_scale'] = 1	
        # parameter_set['TTI'] = 2
        parameter_set['vegcf'] = 0.333427
        parameter_set['WHC'] = 0.133171
        parameter_set['mwf'] = 0.615285
        parameter_set['sfcf'] = 1.052191
        parameter_set['sfcf_scale'] = 0.742628
        parameter_set['masswasting'] = True
        # parameter_set['path_static'] = f"staticmaps_Landwasser_SoilThicknessriver0.2.nc"
        parameter_set[parname] = k
        # parameter_set[parname] = f"staticmaps_Landwasser_{k}.nc"
        model_dics[f"{RUN_NAME}_{k}"] = parameter_set.copy()
            
    #
    t0 = time.time()
    wflow = RunWflow_Julia(
        ROOTDIR = EWC_ROOTDIR,
        PARSETDIR =join(EWC_ROOTDIR,"experiments"),
        BASIN = basin,
        RUN_NAME = 'aug2023test',
        START_YEAR = START_YEAR,
        END_YEAR = END_YEAR,
        CONFIG_FILE = "sbm_config_CH_orig.toml",
        RESOLUTION = resolution ,
        YEARLY_PARAMS=False,
        CONTAINER="",
        NC_STATES = None)
    
    # for year in range(2002,2004):
    #     forcing_name = f"wflow_MeteoSwiss_{wflow.RESOLUTION}_{basin}_{year-2}_{year}.nc"
    #     wflow.generate_MS_forcing_if_needed(forcing_name)
    forcing_name = f"wflow_MeteoSwiss_{wflow.RESOLUTION}_{basin}_{START_YEAR-1}_{END_YEAR}.nc"
    # forcing_name = '/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV/data/input/pettest_Landwasser.nc'
    staticmaps_name = f"staticmaps_{wflow.RESOLUTION}_{basin}_feb2024.nc"
    # staticmaps_name = 
    wflow.generate_MS_forcing_if_needed(forcing_name)
    
    # for year in range(2001,2022):
    #     print(year)
    #     # forcing_name = f"wflow_MeteoSwiss_{wflow.RESOLUTION}_{basin}_{year-1}_{year}.nc"
    #     # wflow.START_YEAR = year-1
    #     # wflow.END_YEAR = year
    #     # wflow.generate_MS_forcing_if_needed(forcing_name)
    #     forcing_name = f"wflow_MeteoSwiss_{wflow.RESOLUTION}_{basin}_{year-2}_{year}.nc"
    #     print(forcing_name)
    #     wflow.START_YEAR = year-1
    #     wflow.END_YEAR = year
    #     wflow.generate_MS_forcing_if_needed(forcing_name)

    # fullname = "/home/pwiersma/scratch/Data/ewatercycle/wflow_Julia_forcing/wflow_MeteoSwiss_1000m_Riale_di_Calneggia_2000_2002.nc"
    # fullname = "/home/pwiersma/scratch/Data/ewatercycle/wflow_Julia_forcing/"+forcing_name

    # forcing_folder = "/home/pwiersma/scratch/Data/ewatercycle/wflow_Julia_forcing/"
    # !rm *

    wflow.check_staticmaps(staticmaps_name)
    
    # sm = xr.open_dataset(join(wflow.ROOTDIR,'wflow_staticmaps',staticmaps_name))
    # sm['SoilThickness'] =sm['SoilThickness'] * 0 +2000
    # sm['SoilThickness'] = xr.where(sm['wflow_river'] ==1,200,sm['SoilThickness'])
    # sm['SoilThickness'] = xr.where(sm['wflow_dem']<1800,300,sm['SoilThickness'])

    # sm['SoilThickness'].plot()
    # sm['InfiltCapSoil'] = xr.where(sm['wflow_river'] ==1,300,sm['InfiltCapSoil'])
    
    # new_file = join(wflow.PARSETDIR,'data/input',
    #                 'staticmaps_Landwasser_KsatVerRF.nc')
    
    # sm['wflow_river'] = xr.where(sm['wflow_streamorder'] ==2,1,sm['wflow_river'])
    # sm['wflow_riverwidth'] = xr.where(sm['wflow_streamorder'] ==2,3,sm['wflow_riverwidth'])
    # sm['wflow_riverlength'] = xr.where(sm['wflow_streamorder'] ==2,50,sm['wflow_riverlength'])
    # sm['RiverDepth'] =  xr.where(sm['wflow_streamorder'] ==2,1,sm['RiverDepth'])
    # sm['RiverSlope'] =  xr.where(sm['wflow_streamorder'] ==2,0.13,sm['RiverSlope'])

    # import numpy as np
    # from scipy.ndimage import gaussian_filter
    # np.random.seed(1)
    
    # # Define the size of your field
    # size = sm['wflow_dem'].data.shape
    
    # data = sm['KsatVer'].copy()
    # plt.figure()
    # data.plot()
    # scale= 0.3
    
    # # Generate a 2D array of random numbers
    # rf = np.random.normal(scale = scale ,size=size, loc = 1)
    
    # # Apply a 2D Gaussian filter to introduce spatial correlation
    # rf = gaussian_filter(rf, sigma=2)
    # plt.figure()
    # plt.imshow(rf)
    # plt.colorbar()

    # adjusted = data + rf
    # plt.figure()
    # adjusted.plot()

    # sm['KsatVer'] = adjusted
    
    # sm.to_netcdf(new_file)
    # wflow.STATICMAPS_FILE = new_file
    
    # # wflow.load_state(parsetname)
    wflow.load_model_dics(model_dics)
    wflow.adjust_config()    
    
    # forcing= xr.open_dataset(wflow.MODEL_DICS[f"{RUN_NAME}_{k}"]['full_forcing_path'])
    # forcing['pet'] *=0
    # new_forcingfile = join(wflow.PARSETDIR,"data/input","pettest_Landwasser.nc")
    # forcing.to_netcdf(new_forcingfile)
    # wflow.MODEL_DICS[f"{RUN_NAME}_{k}"]['config']['input']['path_forcing'] = new_forcingfile
    # forcing= xr.open_dataset(wflow.MODEL_DICS['Landwasser_alltest_-1']['full_forcing_path']).sel(time = '2008')
    # pr = forcing.pr
    # sm = xr.open_dataset(wflow.MODEL_DICS[f"{RUN_NAME}_{k}"]['full_staticmaps_path'])
    
    # dem = sm['wflow_dem'].
    # low_mask = dem<2000
    
    # wflow.create_single_model('default_test9')
    wflow.create_models()
    initialize_time = time.time()-t0
    print(f"Initialization takes {initialize_time} seconds")

    # wflow.series_runs(wflow.degree_day_runs,  test = False )
    wflow.series_runs(wflow.standard_run, test = False)
    # wflow.parallel_runs(wflow.standard_run, test = False)
    wflow.finalize_runs()
    # wflow.save_state(list(model_dics.keys())[0],parsetname)
    wflow.load_stations([basin])
    wflow.load_Q()
    wflow.stations_combine()
    wflow.station_OFs(skip_first_year=True)
    # wflow.standard_plot(add_snow = True, add_rain = False, skip_first_year = True)
    # wflow.standard_plot(add_rain =True, skip_first_year = True)

    # plt.show()
    time1 = time.time()-  time0

    print(f"Years altogether takes {time1} seconds ({time1/60}) minutes")

# wflow.standard_plot(save_figs = "/home/tesla-k20c/data/pau/ewc_output/Figures/Hydrographs/Julia_csnowtest/")


#%%
# year = 2018
# months = [3,4,5,6,7,8,9,10,11,12]
# timeslice = slice(f"{year}-{months[0]}",f"{year}-{months[-1]}")
timeslice = slice(f"{START_YEAR-1}-10",f"{END_YEAR}-07")
Q = wflow.stations[basin].combined
obs = Q['obs'][timeslice]

f1,ax1 = plt.subplots(figsize = (10,5))
obs.plot(ax = ax1, linestyle = 'dashed',color = 'black')
for col in Q.columns:
    if col =='obs':
        continue
    sim = Q[col][timeslice]
    sim.plot(ax = ax1,label = col)

ax1.legend()
ax1.set_ylabel("Discharge [m3/s]")

#%% plot SWE 
C=SnowClass(basin)
SWE_dic  = {}

for key in wflow.MODEL_DICS.keys():    
    dem = xr.open_dataset(join("/home/pwiersma/scratch/Data/ewatercycle"
                                ,'wflow_staticmaps',f'staticmaps_{wflow.RESOLUTION}_{basin}_feb2024.nc'))['wflow_dem']

    C.mask =dem
    output_file =os.path.join(wflow.MODEL_DICS[key]['outfolder'], wflow.MODEL_DICS[key]['config']['output']['path'])
    C.load_SWE_julia(output_file, downscale=True,start_year = START_YEAR,end_year = END_YEAR)
    SWE_dic[key] = C.SWE.rename(dict(longitude = 'lon', latitude = 'lat'))

plt.figure(figsize = (10,5))
for key in wflow.MODEL_DICS.keys():
    SWE_dic[key].sum(dim = ['lat','lon']).sel(
        time=slice(f"{START_YEAR}-10-01",f"{START_YEAR +2}-09-30")).plot(label = key)
    plt.title('Total SWE')
    plt.legend()

timeslice =slice(f"{START_YEAR}-10-01",f"{START_YEAR +2}-09-30")
SWE_filtered1 = SWE_dic['Dischma_mwfets_0.01'].sel(time=timeslice)
SWE_filtered2 = SWE_dic['Dischma_mwfets_1.0'].sel(time=timeslice)
print(SWE_filtered.mean(dim = ['lat','lon']).mean())

SWE_filtered1.mean(dim = 'time').plot()

(SWE_filtered1-SWE_filtered2).mean(dim = ['time']).plot()
#%%
# # var = 'satwaterdepth'
# variables = ['satwaterdepth', 'runoff', 'ustoredepth']
# fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(8, 2 * len(variables)))

# for ax, var in zip(axes, variables):
#     for key in wflow.MODEL_DICS.keys():
#         Synobs_f = f"/home/pwiersma/scratch/Data/ewatercycle/experiments/data/output_{key}/output_{key}.csv"
#         Synobs_scalars = pd.read_csv(Synobs_f, index_col=0, parse_dates=True)
#         Synobs_scalars[var].plot(ax=ax, label=key)
#     ax.legend()
#     ax.set_title(var)
#     ax.grid()

# plt.tight_layout()
# plt.show()

# #%% catchment-wide scalars
# suffix = f"{RUN_NAME}_False"
# Synobs_f = f"/home/pwiersma/scratch/Data/ewatercycle/experiments/data/output_{suffix}/output_{suffix}.csv"
# Synobs_scalars = pd.read_csv(Synobs_f, index_col=0, parse_dates=True)
# # for Synobs_scalars in [Synobs_scalarsobs,Synobs_scalarssim]:
# Syearly = Synobs_scalars.resample('Y').sum()
# Syearly['Q'] = Syearly['Q'] * 1000 * 86400 / (46*1e6)
# # Syearly['Pmean'] *= parameter_set['rfcf']
# Syearly['Pmean'] *= 0.01
# Syearly['SWE'] = Syearly['snow'] + Syearly['snowwater']
# fluximbalance = Syearly['rainfallplusmelt'] - Syearly['et'] - Syearly['Q']

# satimbalance = Synobs_scalars['satwaterdepth'].resample('Y').first().diff()
# unsatimbalance = Synobs_scalars['ustoredepth'].resample('Y').first().diff()
# snowimbalance = Synobs_scalars['snow'].resample('Y').first().diff()

# total_WB = fluximbalance + satimbalance + unsatimbalance + snowimbalance

# plt.figure()
# plt.plot(satimbalance,label = 'satimbalance')
# plt.plot(unsatimbalance,label = 'unsatimbalance')
# plt.plot(fluximbalance,label = 'imbalance fluxes')
# plt.plot(snowimbalance,label = 'snowimbalance')
# plt.plot(total_WB,label = 'total WB')
# plt.axhline(0,color = 'black',linestyle = 'dashed')
# plt.grid()
# plt.legend()
# #%% daily gridded scalars
# def m3s_to_mm(daily_m3s, area):
#     # Convert m3/s to mm/day
#     daily_mm = daily_m3s * 1000 * 86400 / (area*1e6)
#     return daily_mm

# def mm_to_m3s(daily_mm, area):
#     # Convert mm/day to m3/s
#     daily_m3s = daily_mm *1e-3 * (1/86400) * (area*1e6)
#     return daily_m3s
# def m3d_to_mm(daily_m3d, area):
#     # Convert m3/day to mm/day
#     daily_mm = daily_m3d * 1000 / (area*1e6)
#     return daily_mm
# dem =  xr.open_dataset(join("/home/pwiersma/scratch/Data/ewatercycle"
#                                 ,'wflow_staticmaps',f'staticmaps_{wflow.RESOLUTION}_{basin}_feb2024.nc'))['wflow_dem']
# dem = dem.sortby('lat',ascending = True)
# area = 0.591
# ncfile = f"/home/pwiersma/scratch/Data/ewatercycle/experiments/data/output_{suffix}/output_{suffix}.nc"
# output= xr.open_dataset(ncfile).drop_dims('layer')
# output['SWE'] = output['snow'] + output['snowwater']
# output = output.sel(time = slice(f"{END_YEAR}-02",f"{END_YEAR}-09"))
# for var in output.data_vars:
#     if var in ['to_river_ssf','ssf','ssfin']:
#         output[var] = m3d_to_mm(output[var],area)
#     elif var in ['Q','Qin_land','Qin_river','Q_river','inwater_river','inwater_land','to_river_land']:
#         output[var] = m3s_to_mm(output[var],area)
#     # plt.figure()
#     # output[var].mean(dim = 'time').plot()

# #for several grid cells, plot in several subplots the evolutions of satwaterdepth and ustoredepth
# xi = [1,4,8]
# yi = [10,5,3]
# import seaborn as sns
# sns.set(style="whitegrid")

# # Define color palette and linestyles
# colors = sns.color_palette("tab10")
# linestyles = ['-', '--', '-.', ':']

# # Loop through each grid cell
# for i in range(len(xi)):
#     output_i = output.isel(lon=xi[i], lat=yi[i])
#     plt.figure(figsize=(12, 8))
   
#     # Calculate monthly differences
#     monthly_unsat_diff = output_i['ustoredepth'].resample(time='M').first().diff(dim='time')
#     monthly_sat_diff = output_i['satwaterdepth'].resample(time='M').first().diff(dim='time')
#     monthly_swe_diff = output_i['SWE'].resample(time='M').first().diff(dim='time')
    
#     # Plot the data
#     # output_i['runoff'].resample(time='M').sum().plot(label='Runoff', color=colors[0], linestyle=linestyles[1])
#     output_i['rainfallplusmelt'].resample(time='M').sum().plot(label='Rainfall Melt', color=colors[1], linestyle=linestyles[0])
#     monthly_unsat_diff.plot(label='Unsaturated Storage', color=colors[2], linestyle=linestyles[2])
#     monthly_sat_diff.plot(label='Saturated Storage', color=colors[3], linestyle=linestyles[2])
#     monthly_swe_diff.plot(label='SWE', color=colors[4], linestyle=linestyles[2])
#     output_i['actevap'].resample(time='M').sum().plot(label='Evapotranspiration', color=colors[5], linestyle=linestyles[1])
#     # output_i['leakage'].resample(time='M').sum().plot(label='Leakage', color=colors[6], linestyle=linestyles[1])

#     # Lateral fluxes
#     output_i['ssf'].resample(time='M').sum().plot(label='SSF', color=colors[7], linestyle=linestyles[1])
#     output_i['ssfin'].resample(time='M').sum().plot(label='SSF In', color=colors[8], linestyle=linestyles[0])
#     output_i['to_river_ssf'].resample(time='M').sum().plot(label='To River SSF', color=colors[9], linestyle=linestyles[2])
#     # output_i['Qin_land'].resample(time='M').sum().plot(label='Qin Land', color=colors[0], linestyle=linestyles[0])
#     output_i['Qin_river'].resample(time='M').sum().plot(label='Qin River', color=colors[5], linestyle=linestyles[0])
#     output_i['Q_river'].resample(time='M').sum().plot(label='Q River', color=colors[2], linestyle=linestyles[1])
#     # output_i['Q_land'].resample(time='M').sum().plot(label='Q Land', color=colors[3], linestyle=linestyles[1])
#     output_i['inwater_river'].resample(time='M').sum().plot(label='Inwater River', color=colors[3], linestyle=linestyles[0])
#     output_i['inwater_land'].resample(time='M').sum().plot(label='Inwater Land', color=colors[4], linestyle=linestyles[0])
#     # output_i['to_river_land'].resample(time='M').sum().plot(label='To River Land', color=colors[5], linestyle=linestyles[2])

#     # Add legend, title, and labels
#     plt.legend(loc='upper right')
#     plt.title(f"Grid cell {xi[i]},{yi[i]} \n Elevation = {dem.isel(lon=xi[i], lat=yi[i]).values}")
#     plt.xlabel('Time')
#     plt.ylabel('Values')
#     plt.grid(True)

#     # Show the plot
#     plt.show()



# #%%
# # new_simstore = Synobs_scalarssim['ustoredepth'] +(Synobs_scalarssim['et']-Synobs_scalarsobs['et']).cumsum()

# suffix = f"{RUN_NAME}_True"
# Synobs_f = f"/home/pwiersma/scratch/Data/ewatercycle/experiments/data/output_{suffix}/output_{suffix}.csv"
# Synobs_scalarsobs= pd.read_csv(Synobs_f, index_col=0, parse_dates=True)

# suffix = f"{RUN_NAME}_False"
# Synobs_f = f"/home/pwiersma/scratch/Data/ewatercycle/experiments/data/output_{suffix}/output_{suffix}.csv"
# Synobs_scalarssim = pd.read_csv(Synobs_f, index_col=0, parse_dates=True)

# plt.figure()
# Synobs_scalarsobs['ustoredepth'].plot(label = 'obs_unsat')
# Synobs_scalarssim['ustoredepth'].plot(label = 'sim_unsat')

# # new_simstore.plot(label = 'sim_unsat - obs_et')
# Synobs_scalarsobs['satwaterdepth'].plot(label = 'obs_sat')
# Synobs_scalarssim['satwaterdepth'].plot(label = 'sim_sat')
# plt.legend()

# # plt.figure()
# # Synobs_scalarsobs['rainfallplusmelt'].plot(label = 'obs_rainfallplusmelt')
# # Synobs_scalarssim['rainfallplusmelt'].plot(label = 'sim_rainfallplusmelt')
# # plt.legend()

# # plt.figure(figsize = (6,2))
# # Synobs_scalarsobs['et'].plot(label = 'obs_et')
# # Synobs_scalarssim['et'].plot(label = 'sim_et')
# # plt.legend()

# # plt.figure(figsize = (6,2))
# # Synobs_scalarsobs['interception'].plot(label = 'obs_incterception')
# # Synobs_scalarssim['interception'].plot(label = 'sim_interception')
# # plt.legend()

# # C = SnowClass(basin)
# # runoff = C.load_OSHD(resolution = resolution).rename(dict(x = 'lon',y = 'lat'))['romc_all']
# # dem =  xr.open_dataset(join("/home/pwiersma/scratch/Data/ewatercycle"
# #                                 ,'wflow_staticmaps',f'staticmaps_{wflow.RESOLUTION}_{basin}_feb2024.nc'))['wflow_dem']
# # runoff = xr.where(np.isnan(dem),np.nan,runoff)

# # plt.figure()
# # plt.figure()
# # Synobs_scalars['rainfallplusmelt'].plot()
# # runoff.mean(dim = ['lat','lon']).sel(time = '2013').plot()


# #%% Snow analysis
# def Benchmark_efficiency(obs, sim, BE):
#     upper = np.nansum((obs-sim)**2)
#     lower = np.nansum((obs - BE)**2)
#     return 1-upper/lower
    



   
# # for station in C.station_short_names:
# #     for year in range(START_YEAR+1,END_YEAR+1):
# #         f1,ax1 = plt.subplots(figsize=(10,5))
# #         timeslice = slice(f"{year-1}-10",f"{year}-07")
# #         obs = C.HSSWE[station]['SWE'][timeslice]    
# #         jonas2009 = C.HSSWE[station]['SWE_jonas2009'][timeslice]
# #         OSHD = C.HSSWE[station]['OSHD'][timeslice]
# #         if np.all(np.isnan(obs.values.astype(float))):
# #             continue
# #         BE = C.HSSWE[station]['SWE_DOYcorrected'][timeslice]
# #         obs.plot(ax = ax1,label = 'SWE obs', color = 'black',linestyle = 'dashed')
# #         OSHD.plot(ax = ax1, label = 'OSHD', color = 'grey')
# #         # jonas2009.plot(ax = ax1,label = 'SWE_jonas2009', color = 'black',linestyle = 'dashed', alpha = 0.3)
        
# #         # BE.plot(ax = ax1, label = 'DOY benchmark')
# #         for key in wflow.MODEL_DICS.keys():
# #             sim = C.HSSWE[station][key][timeslice]
# #             BE_OF = Benchmark_efficiency(obs.values, sim.values, BE.values)
# #             # sim.plot(ax = ax1,label = f"{key}_BE = {np.round(BE_OF,1)}")
# #             sim.plot(ax = ax1,label = f"{key}")

# #         # C.HSSWE[station]['SWE'][timeslice].plot(ax = ax1,label = 'obs', color = 'black',linestyle = 'dashed')

# #         # if station in C.SWE_stations:
# #         #     (C.HSSWE[station]['HNW_1D'][timeslice]*10).plot(ax = ax1, label = 'Manual SWE obs',
# #         #                                                 linestyle = '', marker = 'x',color = 'black')
# #         ax1.set_title(f"{station} SWE evaluation")
# #         ax1.legend()
# #         ax1.grid()
# #         ax1.set_ylabel('SWE')
# #         plt.show()
    
#     # NDSI = xr.open_dataset("/home/pwiersma/scratch/Data/SnowCover/MOD10A_2000_2022_1km_Jonschwil.nc")['NDSI']
    
# #%%
# for year in range(START_YEAR+1,END_YEAR+1):
#     timeslice = slice(f"{year-1}-10",f"{year}-07")
#     bands = [500,1000,1500,2000,2500,3000]
#     f1,axes = plt.subplots(1,5,sharey = True,figsize = (15,5))
#     plt.subplots_adjust(wspace = 0.05)
#     for i in range(len(bands)-1):
#         b0 = bands[i]
#         b1 = bands[i+1]
#         elevation_mask = xr.where((dem>=b0) & (dem<b1),1,0)
#         res = int(resolution[:-1])
#         area = (np.sum(elevation_mask) * res).item()
#         print(area)
#         if np.all(elevation_mask ==0):
#             print('no elevations in this band')
#             continue

        
#         colors = ['tab:red','tab:blue']
#         alphas = [0.2,0.5]
        
#         # f1,ax1 = plt.subplots(figsize=(10,5))
#         ax = axes[i]

#         for j,key in enumerate(wflow.MODEL_DICS.keys()):    
#             SWE = SWE_dic[key]

#             yearsim = SWE.sel(time = timeslice)
#             yearsim_mass = yearsim * res*res* 0.001
#             # C = evaldic[key]['C']
#             # OSHD = OSHD_whole.sel(time = timeslice)
#             # OSHD_mass = OSHD*res*res*0.001 #tom3
            
#             # FSM = FSM_whole.sel(time = timeslice)
#             # FSM_mass = FSM*res*res*0.001 #tom3
            
#             # minmask = OSHD.min(dim = 'time')<0
            

#             sims = yearsim_mass.where((elevation_mask == 1)&(minmask==False)).sum(dim = ['lat','lon']).to_pandas()
#             # obs1 = OSHD_mass.where((elevation_mask == 1)&(minmask==False)).sum(dim = ['lon','lat']).to_pandas()
#             # obs2 = FSM_mass.where((elevation_mask == 1)&(minmask==False)).sum(dim = ['lon','lat']).to_pandas()
#             # obs *= area/1000 #convert to m3
            
#             # sims = xr.where(minmask,np.nan,sims)
#             # obs = xr.where(minmask,np.nan,obs)

#             # if j ==0:
#             #     obs1.plot(ax = ax,label = 'Reference (OSHD)', color = 'black',linestyle = 'dashed',zorder = 100)
#             #     obs2.plot(ax = ax,label = 'Reference (FSM)', color = 'black',linestyle = 'dotted',zorder = 100)

#             # variables = pd.Index(sims.columns)
            
#             # legend_flag = 0
#             # simcols = variables[variables != 'syn_obs']
#             # variables = variables[variables!= 'spatial_ref']
            
#             # sims = sims[timeslice]
            
            
#             # for s in simcols:
#             sim = sims
#             linestyle = 'solid'
#             alpha = 1
#             # color = blues(k)
#             # if  legend_flag ==0:
#             #     # label = f'Cluster {k} ensemble'
#             #     label = key
#             #     legend_flag +=1
#             # else:
#             #     label = '_noLegend'
#             # sim *= area/1000
#             sim.plot(ax = ax,
#                         label = key,
#                         # color = colors[i],
#                         # alpha = alphas[i],
#                         linestyle = linestyle)
                
#         ax.set_title(f"{b0}-{b1}")
#         if i ==1:
#             ax.legend()
#         ax.grid()
#         ax.set_ylabel('SWE [m3]')
#     f1.suptitle(f"{basin}\n OSHD+FSM comparison per elevation band",y = 1.01)
#     # f1.savefig(join(FIGDIR,f"OSHD_bands/{basin}_{year}.png"))
#     plt.show()

# #%% scatterplot for hock model
# for year in range(START_YEAR,END_YEAR+1):
#     date = f"{year}-04-01"
#     obs = OSHD_whole.sel(time = date)
#     obs = xr.where(~np.isnan(C.mask),obs,np.nan)
#     keys = model_dics.keys()
#     f1,axes = plt.subplots(1,len(keys),figsize = (15,3))
#     for i,key in enumerate(keys):
#         sim = SWE_dic[key].sel(time = date)
#         ax = axes[i]
#         ax.scatter(obs,sim)
#         ax.set_xlim(left = 0)
#         ax.set_ylim(bottom = 0)
#         ax.set_xlabel('OSHD')
#         ax.set_ylabel('Model')
#         r2 = he.r_squared(obs.data.flatten(),sim.data.flatten())
#         ax.set_title(f"r2 = {np.round(r2,2)}")


            
            
# # %%
# f1,ax1 = plt.subplots()
# # Synobs_scalars['ustoredepth'].diff().plot(label= 'unsat_obs',ax = ax1, color = 'blue', linestyle = 'dashed')
# # Synobs_scalars['satwaterdepth'].diff().plot(label = 'sat_obs',ax = ax1, color = 'red', linestyle = 'dashed')
# # self.daily_scalars['ustoredepth']['posterior'].median(axis = 1).diff().plot(label = 'unsat_sim',ax = ax1,color = 'blue',linestyle = 'solid')
# # self.daily_scalars['satwaterdepth']['posterior'].median(axis = 1).diff().plot(label = 'sat_sim',ax = ax1,color = 'red',linestyle = 'solid')
# (Synobs_scalars['Q']*-1).plot(label = 'Q_obs',ax = ax1,color = 'black',linestyle = 'dashed')
# (self.daily_scalars['Q']['posterior'].median(axis = 1)*-1).plot(label = 'Q_sim',ax = ax1,color = 'black',linestyle = 'solid')
# ax1.legend()
# # %%



# fold = xr.open_dataset('/home/pwiersma/scratch/Data/ewatercycle/experiments/data/input/wflow_MeteoSwiss_1000m_Dischma_2014_2020.nc')

# fold['pr'] *= 0
# fold['tas'] *= 0
# fold['tas'] += 10

# frain = fold.copy(deep = True)
# frain['pr'][500,:,:,] = 200
# frain['pr'][1000,:,:,] = 500
# frain['pr'][1500,:,:,] = 1000
# frain.to_netcdf('/home/pwiersma/scratch/Data/ewatercycle/experiments/data/input/wflow_MeteoSwiss_1000m_Dischma_2014_2020_rain.nc') 


# fsnow = fold.copy(deep = True)
# fsnow['pr'][499,:,:,] = 200
# fsnow['pr'][999,:,:,] = 500
# fsnow['pr'][1499,:,:,] = 1000
# fsnow['tas'][499,:,:,] = -20
# fsnow['tas'][999,:,:,] = -20
# fsnow['tas'][1499,:,:,] = -20
# fsnow['tas'][500,:,:,] = 60
# fsnow['tas'][1000,:,:,] = 60
# fsnow['tas'][1500,:,:,] = 60
# fsnow.to_netcdf('/home/pwiersma/scratch/Data/ewatercycle/experiments/data/input/wflow_MeteoSwiss_1000m_Dischma_2014_2020_snow.nc')


# # %%
