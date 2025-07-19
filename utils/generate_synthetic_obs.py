#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:49:49 2023

@author: tesla-k20c
"""
#%%

# import ewatercycle
# import ewatercycle.parameter_sets
# import ewatercycle.analysis
# ewatercycle.CFG.load_file("/home/pwiersma/scratch/Data/ewatercycle/ewatercycle.yaml")
import logging
import warnings
import numpy as np
import pandas as pd
import glob
import os
from os.path import join
import rasterio as rio
from pathos.threading import ThreadPool as Pool
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
import json
# ewatercycle.__version__

os.chdir("/home/pwiersma/scratch/Scripts/Github/ewc")
from SwissStations import *
from SnowClass import *
from RunWflow_Julia import *

from pathlib import Path
import ewatercycle
import numpy as np
from IPython.display import clear_output

os.chdir("/home/pwiersma/")

from ewatercycle_wflowjl.forcing.forcing import WflowJlForcing
from ewatercycle_wflowjl.model import WflowJl
from ewatercycle_wflowjl.utils import get_geojson_locs


from scipy.ndimage import gaussian_filter

os.chdir("/home/pwiersma/scratch/Scripts/Github/ewc")


#%%

def generate_synthetic_obs(synthetic_name,
                           synthetic_params,
                           basin,
                           start_year,
                           end_year,
                           resolution,
                           soil_error_fixedpar = None):
    
    name = synthetic_name
    params = synthetic_params
    START_YEAR = start_year
    END_YEAR = end_year



    time0 = time.time()
    if soil_error_fixedpar!= None:
        RUN_NAME = f"synthetic_{basin}_{name}_fixed{soil_error_fixedpar['par']}_{soil_error_fixedpar['value']}"
    else:
        RUN_NAME = f"synthetic_{basin}_{name}_{resolution}"

    params['DD'] = 'seasonal'
    params['masswasting'] = True
    params['petcf_seasonal'] = True

    
    model_dics = {}
    model_dics[f"{RUN_NAME}"] = params.copy()
            
    #
    t0 = time.time()
    wflow = RunWflow_Julia(
        ROOTDIR = "/home/pwiersma/scratch/Data/ewatercycle/",
        PARSETDIR = "/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV",
        BASIN = basin,
        RUN_NAME = f"synthetic_{name}",
        START_YEAR = START_YEAR,
        END_YEAR = END_YEAR,
        CONFIG_FILE = "sbm_config_CH_orig.toml",
        RESOLUTION = resolution,
        YEARLY_PARAMS=True)
    
    output_folder = join(wflow.ROOTDIR,"experiments/wflow_julia_1000m_2005_2015_JLLRV/synthetic_obs",
                        name)
    
    #TODO place this at the top
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  
    else:
        if (soil_error_fixedpar==None) and (os.path.isfile(join(output_folder,
                             f"Q_{RUN_NAME}.csv"))):
            print(f"Synthetic Q exists already. Break")
            return
        


    forcing_name = f"wflow_MeteoSwiss_{wflow.RESOLUTION}_{basin}_{START_YEAR-1}_{END_YEAR}.nc"
    staticmaps_name = f"staticmaps_{wflow.RESOLUTION}_{basin}_feb2024.nc"
    wflow.generate_MS_forcing_if_needed(forcing_name)
    wflow.check_staticmaps(staticmaps_name)
    wflow.load_model_dics(model_dics)
    wflow.adjust_config()    
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
    # wflow.standard_plot()
    # plt.show()
    time1 = time.time()- time0

    print(f"Years altogether takes {time1} seconds ({time1/60}) minutes")

    #Save Q to station format

    
    Q= wflow.stations[basin].combined[RUN_NAME]
    if soil_error_fixedpar==None:

        Q.to_csv(join(output_folder,f"Q_{RUN_NAME}.csv"))

        #copy SWE output to syntheitc folder
        output_file =os.path.join(wflow.MODEL_DICS[RUN_NAME]['outfolder'], wflow.MODEL_DICS[RUN_NAME]['config']['output']['path'])
        shutil.copy(output_file,join(output_folder,f"SWE_{RUN_NAME}.nc"))

        #Save params to synthetic folder with json dump
        with open(join(output_folder,f"params_{RUN_NAME}.json"), 'w') as fp:
            json.dump(params, fp)
        
        print("All synthetic files written")
    else:
        print("Write soil error adjusted synhtheitc Q file")
        Q.to_csv(join(output_folder,f"Q_{RUN_NAME}.csv"))

def additive_gaussian_noise(Q,scale):
    noise = np.random.normal(size = len(Q),scale = scale,loc = 0)
    noise_corr = gaussian_filter(noise,1)
    return Q + noise_corr

def multiplicative_gaussian_noise(Q,scale):
    noise = np.random.normal(size = len(Q),scale = scale,loc = 1)
    noise_corr = gaussian_filter(noise,1)
    return Q * noise_corr


def generate_obs_error(synthetic_name,
                       basin,
                       obsdic,#or multiplicative
                       ):
    #Load synthetic data
    output_folder = join("/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV/synthetic_obs",
                        synthetic_name)
    RUN_NAME = f"synthetic_{basin}_{synthetic_name}"
    kind = obsdic['kind']
    scale = obsdic['scale']
    
    Q = pd.read_csv(join(output_folder,f"Q_{RUN_NAME}.csv"),index_col = 0, parse_dates = True).squeeze()
    if kind =='multiplicative':
        Q_adjusted = multiplicative_gaussian_noise(Q,scale)
        Q[Q<0] = 0
    elif kind == 'additive':
        Q_adjusted = additive_gaussian_noise(Q,scale)
        Q[Q<0] = 0
    
    Q_adjusted.to_csv(join(output_folder,f"Q_{RUN_NAME}_single{kind}error_{scale}.csv"))

    f1,ax1 = plt.subplots(figsize = (10,5))
    Q[100:300].plot(ax = ax1, label = 'Q_orig')
    Q_adjusted[100:300].plot(ax = ax1, label = 'Q_adjusted')
    ax1.set_ylabel('Q [m3/s]')
    ax1.legend()
    ax1.grid()
    ax1.set_title(f"{kind} noise - {RUN_NAME} - scale = {scale}")
    f1.savefig(join(output_folder,f"Q_{RUN_NAME}_single{kind}error_{scale}.png"))
    
    
    
    # realizations = dict()
    # for i in range(10):
    #     Q = pd.read_csv(join(output_folder,f"Q_{RUN_NAME}.csv"),index_col = 0, parse_dates = True)
    #     orig = Q.columns.item()
    #     Q= Q.rename(columns = {orig:'Q_orig'})
    #     scale = 0.2
    #     Q['noise'] = np.random.normal(size = Q['Q_orig'].size,scale = scale,loc = 0)
    #     Q['noise_corr'] = gaussian_filter(Q['noise'],1)
        
    #     if kind == 'additive':
    #         # Q['noise'] = np.random.normal(size = Q['Q_orig'].size,scale = scale,loc = 0)
    #         # Q['noise_corr'] = gaussian_filter(Q['noise'],1)
    #         # Q['Q_adjusted']= Q['Q_orig'] + Q['noise_corr']
    #         Q['Q_adjusted'] = additive_gaussian_noise(Q['Q_orig'], scale)
    #     elif kind =='multiplicative':
    #         # Q['noise'] = np.random.normal(size = Q['Q_orig'].size,scale = scale,loc = 1)
    #         # Q['noise_corr'] = gaussian_filter(Q['noise'],1)
    #         # Q['Q_adjusted']= Q['Q_orig'] * Q['noise_corr']
    #         Q['Q_adjusted'] = multiplicative_gaussian_noise(Q['Q_orig'], scale)


    #     realizations[i] = Q
    
    # f1,ax1 = plt.subplots(figsize = (10,5))

    # for i in realizations.keys():
    #     print(i)
    #     if i ==0:
    #         label = 'Q_adjusted'
    #     else:
    #         label = '_noLegend'
    #     realizations[i]['Q_adjusted'].iloc[100:500].plot(ax = ax1, color = 'tab:red', alpha = 0.3,
    #                                                      label = label)
    # Q['Q_orig'].iloc[100:500].plot(ax = ax1)#['Q_orig','noise','noise_corr']
    # ax1.grid()
    # ax1.set_title(f"{kind} noise")
    # ax1.set_ylabel('Q [m3/s]')
    # ax1.legend()

    # #%% Old error scaling with rating curve    
    #     # Q.iloc[100:300].plot()
    
    # # scales = [-0.1,0.1,0.2,0.3,0.5]
    # # results = pd.DataFrame(index = Q.index, columns = scales)
    # # results['obs'] = Q
    # # for scale in scales:
            
        
    # Qmin = Q.min().item()
    # Qmax = Q.max().item()

    # x = np.linspace(1,100,100)
    # Qrange = np.linspace(Qmin,Qmax,100)
    
    
    # # # np.random.seed(2)
    # # noise = np.random.normal(size = 100, scale = scale * Q.std())
    # # noise_corr = gaussian_filter(noise,sigma = 30) * Qrange
    
    
    # noise_corr = -np.sin(1.5*np.pi*0.01 * x) * Qrange * scale
    # # plt.plot(x,noise_corr)
    
    # Qrange_adjusted = Qrange + noise_corr
    
    # Q_adjusted = Q*np.nan
    # for i in range(1,Q_adjusted.size):
    #     Q_adjusted.iloc[i,0] = np.interp(Q.iloc[i,0],Qrange,Qrange_adjusted)
    # Q_adjusted[Q_adjusted<0] = 0
    
    # # results.loc[:,scale] = Q_adjusted.values
    #       # 
    # # noise2 = np.random.normal(size = Q.size ,loc = 1, scale = 0.5*scale * Q.std())
    
    # # Q_noised = Q_adjusted *noise2
    
    # # Q_adjusted = Q+noise_corr
    
    # f1, (ax1,ax2) = plt.subplots(2,1,figsize = (10,5))
    # # f1, ax1 = plt.subplots(1,1,figsize = (10,5))

    
    # Q.iloc[:300].plot(ax = ax1,label = 'Synthetic obs')
    # Q_adjusted.iloc[:300].plot(ax = ax1, label = '+noise')
    # ax1.set_ylabel('Q [m3/s]')
    # ax1.legend()
    # # Q_noised.plot(ax= ax1)
    
    # ax2.plot(Qrange,label = 'Original')
    # ax2.plot(Qrange_adjusted, label = '+noise')
    # ax2.set_ylabel('Value range')
    # ax2.legend()
    # f1.suptitle(f"{basin} Synthetic obs error \n scale = {scale}")
    
    # Q_adjusted.to_csv(join(output_folder,f"Q_{RUN_NAME}_obsscale_{scale}.csv"))
    #%%
    
    return 

def generate_soil_error(synthetic_name,
                       basin,
                       soil_error_scale,
                       soil_error_params,
                       shape
                       ):
    output_folder = join("/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV/synthetic_obs",
                        synthetic_name)
    RUN_NAME = f"synthetic_{basin}_{synthetic_name}"
    np.random.seed(1)
    paths = {}
    for par in soil_error_params:
        rf =  gaussian_filter(np.random.normal(size = shape, scale = soil_error_scale, loc = 1), sigma=3)
        filename = join(output_folder,f"{par}_{RUN_NAME}_soilscale_{soil_error_scale}.txt")
        np.savetxt(filename, rf)
        print(f"File {filename} written")
        paths[par] = filename
    return paths
    
    
    
    # plt.figure()
    # Q.hist(bins = 100, alpha = 0.4)
    # Q_adjusted.hist(bins = 100, alpha = 0.4)
    
    








# wflow.standard_plot(save_figs = "/home/tesla-k20c/data/pau/ewc_output/Figures/Hydrographs/Julia_csnowtest/")

# name = 'blabla'
# params  = {}
# generate_synthetic_obs(name,params,
#                        'Landwasser',
#                        2004,2005)
            