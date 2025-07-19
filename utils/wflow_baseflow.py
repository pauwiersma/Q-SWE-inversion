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

os.chdir("/home/pwiersma/scratch/Scripts/Github/ewc")
from SwissStations import *
from SnowClass import *
from RunWflow_Julia import *
# from RunWflow_Julia import *


from pathlib import Path
import ewatercycle
import numpy as np
from IPython.display import clear_output

os.chdir("/home/pwiersma/")
os.chdir("/home/pwiersma/scratch/Scripts/Github/ewc")

# %matplotlib ipympl


#%%


RUN_BASIS = 'baseflow'
resolution = '1000m'
START_YEAR = 1994 #from 1993 for ERA5 variables for now
END_YEAR  =2022

parN = 100

outfolder = "/home/pwiersma/scratch/Data/baseflow_seperation"
today = date.today().strftime("%Y%m%d")
outdir = join(outfolder, today)
Path.mkdir(Path(outdir), exist_ok=True)

pardir = join(outdir, "parameters")
Path.mkdir(Path(pardir), exist_ok=True)

def m3s_to_mm(daily_m3s, area):
    # Convert m3/s to mm/day
    daily_mm = daily_m3s * 1000 * 86400 / (area*1e6)
    return daily_mm
def mm_to_m3s(daily_mm, area):
    # Convert mm/day to m3/s
    daily_m3s = daily_mm *1e-3 * (1/86400) * (area*1e6)
    return daily_m3s
def m3d_to_mm(daily_m3d, area):
    # Convert m3/day to mm/day
    daily_mm = daily_m3d * 1000 / (area*1e6)
    return daily_mm

area = 0.591

basins = [sys.argv[1]]
# basins = ['Dischma']
# basins =   ['Mogelsberg','Dischma','Ova_da_Cluozza','Sitter','Werthenstein',
#         'Sense','Ilfis','Eggiwil','Chamuerabach','Veveyse','Minster','Ova_dal_Fuorn','Alp','Biber','Riale_di_Calneggia',
#     'Chli_Schliere','Allenbach','Jonschwil','Landwasser','Landquart','Rom','Verzasca','Muota']
# 'Kleine_Emme','Riale_di_Pincascia',
time0 = time.time()
for basin in basins:
    print(f"Starting with {basin}")
    RUN_NAME = f"{basin}_{RUN_BASIS}"
      
    model_dics = {}
    
    for ii in range(parN):
        # parameter_set = posterior_parset[np.random.randint(0,len(posterior_parset))]
        # parameter_set = posterior_parset[0]
        # del parameter_set
        parameter_set = {}
        parameter_set['KsatHorFrac'] = np.random.uniform(1,100)
        parameter_set['f'] = np.random.uniform(0.3,2)
        parameter_set['c'] = np.random.uniform(1,10)
        parameter_set['KsatVer'] = np.random.uniform(0.5,1.2)

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
        parameter_set['CV'] = 0.3
        # parameter_set[]

        # parameter_set['DD'] = 'static'
        parameter_set['m_hock'] = 2.5
        parameter_set['r_hock'] = 0.01
        parameter_set['vegcf'] = 0
        parameter_set['WHC'] = 0.2


        parameter_set['TT'] = np.random.uniform(-3,3)
        parameter_set['TTI'] = 2
        # parameter_set['path_static'] = f"staticmaps_Landwasser_SoilThicknessriver0.2.nc"
        # parameter_set[parname] = k
        # parameter_set[parname] = f"staticmaps_Landwasser_{k}.nc"
        model_dics[f"{RUN_NAME}_{ii}"] = parameter_set.copy()
            
    #
    t0 = time.time()
    wflow = RunWflow_Julia(
        ROOTDIR = "/home/pwiersma/scratch/Data/ewatercycle/",
        PARSETDIR = "/home/pwiersma/scratch/Data/ewatercycle/experiments/",
        BASIN = basin,
        RUN_NAME = 'aug2023test',
        START_YEAR = START_YEAR,
        END_YEAR = END_YEAR,
        CONFIG_FILE = "sbm_config_CH_orig.toml",
        RESOLUTION = resolution ,
        YEARLY_PARAMS=False,
        CONTAINER="",
         NC_STATES =None)

    forcing_name = f"wflow_MeteoSwiss_{wflow.RESOLUTION}_{basin}_{START_YEAR-1}_{END_YEAR}.nc"
    # forcing_name = '/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV/data/input/pettest_Landwasser.nc'
    staticmaps_name = f"staticmaps_{wflow.RESOLUTION}_{basin}_feb2024.nc"
    wflow.generate_MS_forcing_if_needed(forcing_name)
    
    wflow.check_staticmaps(staticmaps_name)
        
    wflow.load_model_dics(model_dics)
    wflow.adjust_config()    
    wflow.create_models()
    initialize_time = time.time()-t0
    # print(f"Initialization takes {initialize_time} seconds")

    wflow.series_runs(wflow.standard_run, test = False)
    wflow.finalize_runs()
    wflow.load_stations([basin])
    wflow.load_Q()
    wflow.stations_combine()
    wflow.station_OFs(skip_first_year=True)

    time1 = time.time()-  time0

    # print(f"Years altogether takes {time1} seconds ({time1/60}) minutes")

    scalars = []
    for i in range(parN):
        suffix = f"{RUN_NAME}_{i}"
        simscalars_f = f"/home/pwiersma/scratch/Data/ewatercycle/experiments/data/output_{suffix}/output_{suffix}.csv"
        scalardata = pd.read_csv(simscalars_f, index_col=0, parse_dates=True)
        scalardata['baseflow'] = mm_to_m3s(m3d_to_mm(scalardata['baseflow'],area),area)
        scalars.append(scalardata[['Q','baseflow','Pmean','Tmean']])
        
    # for i in [0]:
    #     plt.figure()
    #     scalars[i]['Q'].plot(label  = 'Q_total')
    #     scalars[i]['baseflow'].plot(label = 'baseflow')
    #     plt.ylabel('Q [m3/s]')
    #     plt.legend()
    #     plt.show()

    # from hydrosignatures import baseflow_index
    # bfi = scalardata['baseflow'].sum()/scalars[i]['Q'].sum()
# 
   
    for i in range(parN):
        suffix = f"{basin}_{i}"
        scalars[i].to_csv(join(outdir,f"baseflow_{suffix}.csv"))
    
    # parnames = list(model_dics[f"{RUN_NAME}_0"].keys())
    parnames = ['KsatHorFrac','f','c','KsatVer','TT']
    pardf = pd.DataFrame(columns = parnames)
    for i in range(parN):
        for par in parnames:
            pardf.loc[i,par] = model_dics[f"{RUN_NAME}_{i}"][par]

    pardf.to_csv(join(pardir,f"parameters_{basin}.csv"))



