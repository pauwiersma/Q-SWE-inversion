#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:37:03 2022

@author: Pau Wiersma

For Wflow Julia

"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import ewatercycle
# import ewatercycle.parameter_sets
# import ewatercycle.analysis
# ewatercycle.CFG.load_from_file("/home/tesla-k20c/ssd/pau/ewatercycle/ewatercycle.yaml")

from SwissStations import *
from SnowClass import *


import logging
import warnings
import numpy as np
import pandas as pd
import glob
import os
from os.path import join
import rasterio as rio
from datetime import datetime

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
import HydroErr as he
import shutil
import math
#import memory_profiler
# ewatercycle.__version__
# from pywflow import WflowBMI

from pathlib import Path
import ewatercycle
EWC_RUNDIR = os.getenv('EWC_RUNDIR', default = '/home/pwiersma/scratch/Scripts/Github/ewc')
ewatercycle.CFG.load_from_file(join(EWC_RUNDIR, 'ewatercycle_config.yaml'))

from ewatercycle_wflowjl.forcing.forcing import WflowJlForcing
from ewatercycle_wflowjl.model import WflowJl,install_wflow
from ewatercycle_wflowjl.utils import get_geojson_locs

from ewatercycle.base.parameter_set import ParameterSet

# import tomli
# import tomli_w
import tomlkit

# from ewatercycle.analysis import hydrograph

# os.chdir("/home/tesla-k20c/ssd/pau/Github/ewc")
# os.chdir("/home/pwiersma/github/ewc")

from sys import settrace, _getframe as gf
def f(frame, event, arg):
        print(gf().f_code.co_name)
        print(frame)
        print(event)
        print('---')
        return f
    
# os.chdir("/home/pwiersma/scratch/Scripts/Github/ewc/utils")
import sys
# sys.path.append("/home/pwiersma/scratch/Scripts/Github/ewc/utils")
# from Hydrograph_lines import *

#settrace(f)

#%%
"""
To add
- pointer to forcing and staticmaps

"""

class RunWflow_Julia:
    def __init__(self,ROOTDIR, 
                 PARSETDIR,  
                 BASIN,
                 CONFIG_FILE,
                 START_YEAR,
                 END_YEAR,
                 RUN_NAME,
                 RESOLUTION = '1000m',
                 YEARLY_PARAMS = False,
                 SYNTHETIC_OBS = None  ,
                 CONTAINER= None,
                 NC_STATES = None,
                 ):
        self.ROOTDIR = ROOTDIR 
        self.PARSETDIR = PARSETDIR
        self.BASIN = BASIN
        self.RUN_NAME = RUN_NAME
        # self.WFLOW_VERSION = WFLOW_VERSION
        self.RESOLUTION = RESOLUTION 
        self.CONFIG_FILE =os.path.join(PARSETDIR,CONFIG_FILE)
        self.START_YEAR = START_YEAR
        self.END_YEAR = END_YEAR
        self.remove_sm = []
        self.YEARLY_PARAMS = YEARLY_PARAMS
        self.SYNTHETIC_OBS = SYNTHETIC_OBS #needs to be full directory 
        self.CONTAINER = CONTAINER
        self.NC_STATES = NC_STATES


        def compute_grid_area():
            lat_deg = float(self.actualres)
            lon_deg = float(self.actualres)
            lat_km = 111.32 * lat_deg
            lon_km = 111.32 * lon_deg * math.cos(math.radians(self.LATITUDE))
            grid_area = lat_km * lon_km
            return grid_area

        self.actualres = int(self.RESOLUTION[:-1]) *8.333e-06 
        self.LATITUDE = SwissStation(self.BASIN).lat
        self.grid_area = compute_grid_area()
        # self.OUTPUT_DIR = join(OUTDIR,'custom',RUN_NAME)
        # if not os.path.isdir(self.OUTPUT_DIR):
        #     os.mkdir(self.OUTPUT_DIR)
    def load_model_dics(self, MODEL_DICS):
        """One entry for each variable, plus a directory for external_forcing
        Should include also the cfg and its path"""
        self.MODEL_DICS = MODEL_DICS
    def adjust_config(self):
        for name in self.MODEL_DICS.keys():
            if self.CONTAINER in [None,'']:
                infolder = join(self.PARSETDIR,"data/input")
            else:
                infolder = self.CONTAINER
            # infolder = join(self.PARSETDIR,"data/input")
            outfolder =   join(self.PARSETDIR,f"data/output_{name}")
            if not os.path.isdir(outfolder):
                os.mkdir(outfolder)
                    
            # Adjust config file
            with open(self.CONFIG_FILE, mode = 'rb') as fp :
                config = tomlkit.load(fp)
                
            if 'FORCING_FILE' in dir(self):
                config['input']['path_forcing'] = self.FORCING_FILE
            if 'STATICMAPS_FILE' in dir(self):
                config['input']['path_static'] = self.STATICMAPS_FILE
            if 'instatepath' in dir(self):
                config['state']['path_input'] = self.instatepath
                config['model']['reinit'] = False
                print(self.instatepath)
            
            config['starttime'] = f"{self.START_YEAR-1}-10-01T00:00:00"
            config['endtime'] = f"{self.END_YEAR}-09-30T00:00:00"
            config['dir_output'] = f"data/output_{name}"
            config['csv']['path'] = f"output_{name}.csv"
            config['csv']['column'].append(dict(header = 'Pmean',parameter = 'vertical.precipitation',reducer='mean'))
            config['netcdf']['path'] = f"output_scalar_{name}.nc"
            config['output']['path'] = f"output_{name}.nc"
            config['state']['path_output'] = f"outstates_{name}.nc"
            config['loglevel'] = 'warn'
            config['silent'] = True
            config['model']['kin_wave_iteration'] = False
            # config['input']['vertical']['cfmax'] = 'Cfmax'

            for par, value in self.MODEL_DICS[name].items():
                if par =='KsatHorFrac':
                    config['input']['lateral']['subsurface']['ksathorfrac']['value'] = value
                if par =='f':
                    config['input']['vertical']['f']['scale'] = value
                if par =='Cfmax':
                    config['input']['vertical']['cfmax']['value'] = value
                if par =='sfcf':
                    config['input']['vertical']['sfcf']['value'] =value
                    config['input']['vertical']['sfcf_value'] = value
                if par =='sfcf_scale':
                    config['input']['vertical']['sfcf_scale'] = value
                if par == 'm_hock':
                    config['input']['vertical']['m_hock'] = value
                if par == 'r_hock':
                    config['input']['vertical']['r_hock'] = value
                if par =='rfcf':
                    config['input']['vertical']['rfcf']['value'] = value
                if par =='CV':
                    config['input']['vertical']['cv']['value'] = value
                if par =='mwf':
                    config['input']['lateral']['snow']['mwf'] = value
                if par =='KsatVer':
                    config['input']['vertical']['kv_0']['scale'] = value
                if par =='SoilThickness':
                    config['input']['vertical']['soilthickness']['scale'] = value
                if par =='thetaR':
                    config['input']['vertical']['theta_r']['scale'] = value
                if par =='thetaS':
                    config['input']['vertical']['theta_s']['scale'] = value
                if par =='c':
                    config['input']['vertical']['c']['scale'] = value
                if par == 'TT':
                    config['input']['vertical']['tt']['offset'] = value
                    config['input']['vertical']['ttm']['value'] = value
                if par == 'TTI':
                    config['input']['vertical']['tti']['value'] = value
                # if par =='TTM':
                #     config['input']['vertical']['ttm']['value'] = value
                if par =='WHC':
                    config['input']['vertical']['whc']['offset'] = value
                if par =='vegcf':
                    config['input']['vertical']['vegcf']['value'] = value
                if par == 'path_forcing':
                    config['input']['path_forcing'] = value
                if par == 'path_static':
                    config['input']['path_static'] = value
                if par =='kin_wave': 
                    config['model']['kin_wave_iteration'] = value
                if par == 'loglevel':
                    config['loglevel'] = value
                if par =='silent':
                    config['silent'] = value
                if par =='precip_scaling':
                    config['input']['vertical']['precipitation']['scale'] = value
                if par == 'N':
                    config['input']['lateral']['river']['n']['scale'] = value
                if par =='masswasting':
                    config['model']['masswasting'] = value
                if par =='InfiltCapSoil':
                    config['input']['vertical']['infiltcapsoil']['scale']= value
        
            if self.NC_STATES!=None:
                for var in self.NC_STATES:
                    if var in ['actevap','actleakage','rainfallplusmelt','canopystorage','interception',
                               'satwaterdepth','ustoredepth','e_r','throughfall','runoff','transfer','vwc','zi','ustorelayerdepth','act_thickl']:
                        # if var == 'satwaterdepth':
                        #     var = 'satstore'
                        # elif var == 'ustoredepth':
                        #     var = 'unsatstore'
                        # elif var =='ustorelayerdepth':
                        #     var = 'unsatstore_layer'
                        # elif var =='rainfallplusmelt':
                        #     var = 'rfmelt'
                        # elif var =='actleakage':	
                        #     var = 'leakage'
                        config['output']['vertical'][var] = var
                    elif var in ['q_river','qin_river','inwater_river']:
                        wflowvar = var.split('_')[0]#.lower()
                        config['output']['lateral']['river'][wflowvar] = var
                    elif var in ['q_land','qin_land','inwater_land','to_river_land']:
                        if var == 'to_river_land':
                            wflowvar = 'to_river'
                        # else:
                        #     wflowvar = var 
                        else:
                            wflowvar = var.split('_')[0]#.lower()
                        config['output']['lateral']['land'][wflowvar] = var
                    elif var in ['ssf','ssfin','to_river_ssf']:
                        if var == 'to_river_ssf':
                            wflowvar = 'to_river'
                        else:
                            wflowvar = var
                        config['output']['lateral']['subsurface'][wflowvar] = var

            # if self.NC_STATES:
            #     print("Saving additional states to netcdf")
            #     config['output']['vertical']['actevap'] = 'actevap'
            #     config['output']['vertical']['actleakage'] = 'leakage'
            #     config['output']['vertical']['rainfallplusmelt'] = 'rfmelt'
            #     config['output']['vertical']['canopystorage'] = 'canopystorage'
            #     config['output']['vertical']['interception'] = 'interception'
            #     config['output']['vertical']['satwaterdepth'] = 'satstore'
            #     config['output']['vertical']['ustoredepth'] = 'unsatstore'
            #     config['output']['vertical']['e_r'] = 'canopy_evap_fraction'
            #     config['output']['vertical']['throughfall'] = 'throughfall'
            #     config['output']['vertical']['runoff'] = 'runoff'
            #     config['output']['vertical']['transfer'] = 'transfer'
            #     config['output']['vertical']['vwc'] = 'vwc'
            #     config['output']['vertical']['zi'] = 'zi'
            #     config['output']['vertical']['ustorelayerdepth'] = 'unsatstore_layer'
            #     config['output']['vertical']['act_thickl'] = 'act_thickl'
            #     # config['output']['vertical']['sumlayers'] = 'sumlayers'

            #     config['output']['lateral']['river']['q'] = 'Q_river'
            #     config['output']['lateral']['river']['qin'] = 'Qin_river'
            #     config['output']['lateral']['river']['inwater'] = 'inwater_river'

            #     config['output']['lateral']['land']['q'] = 'Q_land'
            #     config['output']['lateral']['land']['qin'] = 'Qin_land'
            #     config['output']['lateral']['land']['inwater'] = 'inwater_land'
            #     config['output']['lateral']['land']['to_river'] = 'to_river_land'

            #     config['output']['lateral']['subsurface']['ssf'] = 'ssf'
            #     config['output']['lateral']['subsurface']['ssfin'] = 'ssfin'
            #     config['output']['lateral']['subsurface']['to_river'] = 'to_river_ssf'
            #     # config['output']['lateral']['subsurface']['volume'] = 'ssvolume'
            #     # config['output']['lateral']['river']['Q'] = 'q'
            
            config_out = join(self.PARSETDIR,f"sbm_config_{self.BASIN}_{name}.toml")
            
            self.MODEL_DICS[name]['infolder']  = infolder
            self.MODEL_DICS[name]['outfolder'] = outfolder
            self.MODEL_DICS[name]['config']    = config
            self.MODEL_DICS[name]['config_dir'] = config_out
            self.MODEL_DICS[name]['full_staticmaps_path'] = os.path.join(infolder,config['input']['path_static'])
            self.MODEL_DICS[name]['full_forcing_path'] = os.path.join(infolder,self.MODEL_DICS[name]['config']['input']['path_forcing'])
            
            yearly_params_list= [] #weird place to put this, but it works for now

            new_staticmaps = xr.open_dataset(self.MODEL_DICS[name]['full_staticmaps_path'])
            new_forcing = xr.open_dataset(self.MODEL_DICS[name]['full_forcing_path'])#.sel(time = slice(str(config['starttime']),str(config['endtime'])))

            new_staticmaps_fname = f"{self.MODEL_DICS[name]['full_staticmaps_path'][:-3]}_{name}.nc"
            new_forcing_fname = f"{self.MODEL_DICS[name]['full_forcing_path'][:-3]}_{name}.nc"

            if self.YEARLY_PARAMS:
                #load original forcing 
                #load the yearly values for each candidate parameter (sfcf,sfcf_scale, rfcf, TT, tt_scale,DD_min,DD_max,mwf,WHC,vegcf)
                #Make nc variables for sfcf, rfcf, TT, DD_min,DD_max,mwf,WHC,vegcf
                #Adjust sfcf and TT with sfcf_scale ad tt_scale 
                #Make new forcing file with these variables and add it to the config
                #Save to file 
                #Add file to list of files to remove

                #Load forcing
                # orig_forcing_path = self.MODEL_DICS[name]['full_forcing_path']
                # orig_forcing = xr.open_dataset(orig_forcing_path)
                
                #Check which parameter vary yearly 
                for par in ['sfcf','rfcf','TT','DD_min','DD_max','mwf','WHC','vegcf']:
                    if f"{par}_{self.END_YEAR}" in self.MODEL_DICS[name].keys():
                        print("Writing yearly parameter", par)
                        if par =='DD_min':
                            write_par = 'Cfmax'
                        elif par =='DD_max':
                            if 'DD_min' in yearly_params_list:
                                continue
                            else:
                                write_par = 'Cfmax'
                        else:
                            write_par = par
                        yearly_params_list.append(par)
                        # config['input']['forcing'].append(write_par)
                        new_forcing[write_par] = new_forcing['pr'] * 0
                        # orig_staticmaps = xr.open_dataset(self.MODEL_DICS[name]['full_staticmaps_path'])
                        for year in range(self.START_YEAR,self.END_YEAR+1): #no spinup year, it's considered implicit
                            if year ==self.START_YEAR:
                                # if 'instatepath' in dir(self):
                                #     print(self.instatepath)
                                #     print("Skip first year ")
                                #     continue
                                # else: 
                                #default value for spinup year 
                                # print("spinup value")
                                if write_par in ['sfcf','rfcf']:
                                    value = config['input']['vertical'][write_par]['value']
                                elif write_par == 'Cfmax':
                                    value = config['input']['vertical']['cfmax']['value']
                                elif write_par in ['TT','WHC']:
                                    value = config['input']['vertical'][write_par.lower()]['offset']
                                elif write_par =='mwf':
                                    value = config['input']['lateral']['snow']['mwf']
                                    print('value=', value)
                                elif write_par =='vegcf':
                                    value = 0
                                    
                            else:    
                                    
                                value = self.MODEL_DICS[name][f"{par}_{year}"]
                                print(f"{par}_{year}  = {value}")
                                
                                

                                if write_par == 'sfcf':

                                    if 'sfcf_scale' in self.MODEL_DICS[name].keys() :
                                        print('scaling sfcf')
                                        if f'sfcf_scale_{year}' in self.MODEL_DICS[name].keys():
                                            # print('sfcf_scale', year)
                                            sfcf_scale = self.MODEL_DICS[name][f'sfcf_scale_{year}']
                                        else:
                                            # if 'sfcf_scale'
                                            sfcf_scale = self.MODEL_DICS[name][f'sfcf_scale']
                                        if 'scaled_dem' not in dir(self):
                                            dem = new_staticmaps['wflow_dem']
                                            scaled_dem = self.scale_dem(dem,below_avg_to_0 = False)
                                        else:
                                            scaled_dem = self.scaled_dem
                                        sfcf_correctionfield = scaled_dem * (sfcf_scale-1) +1 
                                        value = sfcf_correctionfield * value
                                    
                                                                    
                                    if year == self.END_YEAR:
                                        print(f"ADding {write_par} to toml as forcing")
                                        #Adding the forcing variable to the config file
                                        config['input']['forcing'].append(f"vertical.{write_par}")
                                        
                                        config['input']['vertical'].pop('sfcf')
                                        #make sure sfcf points to the sfcf variable in the frocing file 
                                        config['input']['vertical'].append('sfcf','sfcf')

                                
                                        
                                
                                elif (par == 'TT') :#in self.MODEL_DICS[name].keys():
                                    print("scaling TT yealy")
                                    if 'tt_scale_{year}' in self.MODEL_DICS[name].keys():
                                    # print('tt_scale')
                                        tt_scale = self.MODEL_DICS[name][f'tt_scale_{year}']
                                    else:
                                        tt_scale = self.MODEL_DICS[name][f'tt_scale']
                                    dem = new_staticmaps['wflow_dem']
                                    norm_dem = dem/np.nanmean(dem)
                                    plusmin_dem = xr.where(norm_dem<=1, (1/-norm_dem)+1, norm_dem -1)
                                    #If there is a non-zero tt_scale, it is applied to the scaled demhere 
                                    scaled_tt = plusmin_dem * tt_scale
                                    #Then the TT correction is added to DD and added to the original TT
                                    # Negative tt_scale will make the TT higher at lower elevations
                                    # Positive tt_scale will make the TT higher at higher elevations
                                    value = scaled_tt + value + new_staticmaps['TT'] 
                                    if year == self.END_YEAR:
    
                                        config['input']['forcing'].append(f"vertical.{par.lower()}")
    
                                elif ('DD' in par) and ('DD' in self.MODEL_DICS[name].keys()):
                                    # print('seasonal')
                                    DOY = new_forcing.sel(dict(time=slice(f"{year-1}-10-01", f"{year}-09-30"))).time.dt.dayofyear
                                    DD_min = self.MODEL_DICS[name][f'DD_min_{year}']
                                    DD_max = self.MODEL_DICS[name][f'DD_max_{year}']
                                    value =  (((DD_min + DD_max)/2) + ((DD_max-DD_min)/2)*np.sin((2*np.pi*(DOY-80.25)/365.25)))
                                    if year == self.END_YEAR:
    
                                        config['input']['forcing'].append(f"vertical.cfmax")
                                    config['input']['vertical'].pop('cfmax')
                                    config['input']['vertical'].append('cfmax','Cfmax')
                                    print(config['input']['vertical']['cfmax'])
    
                                else:
                                    # print(par, " uniform parameter")
                                    if year == self.END_YEAR:
                                        if par =='WHC':
                                            config['input']['forcing'].append(f"vertical.whc")
                                        elif par =='vegcf':
                                            config['input']['forcing'].append(f"vertical.vegcf")
                                            config['input']['vertical'].pop('vegcf')
                                            config['input']['vertical'].append('vegcf','vegcf')
                                        elif par =='rfcf':
                                            config['input']['forcing'].append(f"vertical.rfcf")
                                            config['input']['vertical'].pop('rfcf')
                                            config['input']['vertical'].append('rfcf','rfcf')
                                        elif par =='mwf':
                                            config['input']['forcing'].append(f"lateral.snow.{write_par}")
                                            config['input']['lateral']['snow'].pop('mwf')
                                            config['input']['lateral']['snow'].append('mwf','mwf')
                                        
                                
                            new_forcing[write_par].loc[dict(time=slice(f"{year-1}-10-01", f"{year}-09-30"))] = value
                
                #Save to file under new name
                # fraction = int(np.round(np.modf(time.time())[0],4)*10000)
                # fraction = np.modf(time.time())[0]
                # new_forcing_dir = os.path.join(self.MODEL_DICS[name]['infolder'],
                #                                f"{orig_forcing_path[:-3]}_yearly_params_{fraction}.nc") #Make sure we have a random path 
                # orig_forcing.to_netcdf(new_forcing_dir)
                #Adjust forcing path in config file
                # config['input']['path_forcing'] = new_forcing_dir  
                # config['full_forcing_path'] = new_forcing_dir

                # self.remove_sm.append(new_forcing_dir)
            
            if 'sfcf_scale' in self.MODEL_DICS[name].keys():
                if (not 'sfcf_scale' in yearly_params_list) & (not 'sfcf' in yearly_params_list):
                    print("Applying elevation-dependent snowfall correction")
                    # orig_staticmaps = xr.open_dataset(self.MODEL_DICS[name]['full_staticmaps_path'])
                    sfcf_scale = self.MODEL_DICS[name]['sfcf_scale']
                    if 'scaled_dem' not in dir(self):
                        dem = new_staticmaps['wflow_dem']
                        scaled_dem = self.scale_dem(dem,below_avg_to_0 = False)
                    else:
                        scaled_dem = self.scaled_dem
                    scaled_sfcf = scaled_dem * (sfcf_scale-1) +1 
                    #To get low precip low and high precip high elevation: sfcf = 0.8, sfcf_scale = 2
                    #to Get high precip low and low precip high elevation: sfcf = 1.5, sfcf_scale = 0.7

                    
                    # new_staticmaps = orig_staticmaps.copy()
                    new_staticmaps['sfcf'] = scaled_sfcf * config['input']['vertical']['sfcf']['value']
                    # new_staticmaps_fname = f"{config['input']['path_static'][:-3]}_sfcf_scale_{sfcf_scale+np.random.uniform(0.1,1)/1000}.nc"
                
                    
                    # new_staticmaps.to_netcdf(join(infolder,new_staticmaps_fname))
                    # new_staticmaps.close()
                    # config['input']['path_static'] = new_staticmaps_fname
                    # self.remove_sm.append(join(infolder,new_staticmaps_fname))
                    # self.MODEL_DICS[name]['full_staticmaps_path'] = os.path.join(infolder,config['input']['path_static'])
                    try:
                        config['input']['vertical']['sfcf']  = 'sfcf'
                    except:
                        config['input']['vertical']['sfcf']  = 'sfcf'

                
            if 'tt_scale' in self.MODEL_DICS[name].keys():
                if not 'tt_scale' in yearly_params_list:
                    print("Applying elevation-dependent melt temperature correction")
                    tt_scale = self.MODEL_DICS[name]['tt_scale']  # degrees per km
                    TT_offset = config['input']['vertical']['tt']['offset']  # flat offset

                    dem = new_staticmaps['wflow_dem']
                    elev_ref = np.nanmean(dem)
                    lapse_per_meter = tt_scale / 1000.0

                    delta_z = dem - elev_ref
                    scaled_tt = lapse_per_meter * delta_z + TT_offset

                    new_staticmaps['TT'] = new_staticmaps['TT'] + scaled_tt
                    new_staticmaps['TTM'] = new_staticmaps['TTM'] + scaled_tt
                    print("Adjusting also TTM")
                    # orig_staticmaps = xr.open_dataset(self.MODEL_DICS[name]['full_staticmaps_path'])
                    # tt_scale = self.MODEL_DICS[name]['tt_scale']
                    # dem = new_staticmaps['wflow_dem']
                    # norm_dem = dem/np.nanmean(dem)
                    # TT_offset = config['input']['vertical']['tt']['offset']
                
                    
                    # plusmin_dem = xr.where(norm_dem<=1, (1/-norm_dem)+1, norm_dem -1)
                    # # maxi = plusmin_dem.max().item()
                    # # stretched = plusmin_dem*(1/maxi)
                    
            
                    # scaled_tt = plusmin_dem * tt_scale
                    # # new_staticmaps = orig_staticmaps.copy()
                    # new_staticmaps['TT'] = new_staticmaps['TT'] + TT_offset + scaled_tt
                    # new_staticmaps['TTM'] = new_staticmaps['TTM'] + TT_offset + scaled_tt
                    # plt.figure()
                    # new_staticmaps['TT'].plot()
                    # fraction = np.modf(time.time())[0]
                    # new_staticmaps_fname = f"{config['input']['path_static'][:-3]}_tt_scale_{TT_offset}_{tt_scale}_{fraction}.nc"
                    # print(new_staticmaps_fname)
                    # new_staticmaps.to_netcdf(join(infolder,new_staticmaps_fname))
                    # new_staticmaps.close()
                    # config['input']['path_static'] = str(new_staticmaps_fname)
                    # self.remove_sm.append(join(infolder,new_staticmaps_fname))
                    # self.MODEL_DICS[name]['full_staticmaps_path'] = os.path.join(infolder,config['input']['path_static'])
                    try:
                        config['input']['vertical'].pop('tt', None)
                        config['input']['vertical']['tt'] = 'TT'   
                        config['input']['vertical'].pop('ttm', None)
                        config['input']['vertical']['ttm'] = 'TTM'
                        # config['input']['vertical']['tt']['offset'] = 0
                        # config['input']['vertical']['tt']  = 'TT'
                    except:
                        config['input']['vertical'].pop('tt', None)
                        config['input']['vertical']['tt'] = 'TT'   
                        config['input']['vertical'].pop('ttm', None)
                        config['input']['vertical']['ttm'] = 'TTM'
                        # config['input']['vertical']['tt']['offset'] = 0
                        # config['input']['vertical']['tt']  = 'TT'
            
            if 'T_offset' in self.MODEL_DICS[name].keys():
                # Add a constant to the forcing temperature
                T_offset = self.MODEL_DICS[name]['T_offset']
                if T_offset ==0:
                    print("No temperature offset")
                else:
                    print("Adjusting temperature")
                    if not 'T_offset' in yearly_params_list:
                        new_forcing['tas'] += self.MODEL_DICS[name]['T_offset']
                    
            
            if 'DD' in self.MODEL_DICS[name].keys():
                if (not 'DD_min' in yearly_params_list) & (not 'DD_max' in yearly_params_list):
                    # Adding a forcing variable to make the melt factor vary in time 
                    #Load forcing
                    # orig_forcing = self.MODEL_DICS[name]['config']['input']['path_forcing']
                    # forcing = xr.open_dataset(os.path.join(infolder,orig_forcing))
                    #Create new forcing variable similar to others called "Cfmax"
                    #Adjust Cfmax based on DDmin, DDmax and fsca
                    default_cfmax = config['input']['vertical']['cfmax']['value']
                    if self.MODEL_DICS[name]['DD'] == 'static':
                        print('static')
                        DD = default_cfmax
                        new_forcing['Cfmax'] = xr.ones_like(new_forcing['pr']) *DD

                    elif self.MODEL_DICS[name]['DD'] == 'seasonal':
                        print('seasonal')
                        DOY = new_forcing.time.dt.dayofyear
                        DD_min = self.MODEL_DICS[name]['DD_min']
                        DD_max = self.MODEL_DICS[name]['DD_max']
                        DD =  (((DD_min + DD_max)/2) + ((DD_max-DD_min)/2)*np.sin((2*np.pi*(DOY-80.25)/365.25)))
                        new_forcing['Cfmax'] = xr.ones_like(new_forcing['pr']) *DD
                    elif self.MODEL_DICS[name]['DD'] == 'Hock':
                        print('Hock melt model')
                        
                        DOY = new_forcing.time.dt.dayofyear
                        m = self.MODEL_DICS[name]['m_hock']
                        r_hock = self.MODEL_DICS[name]['r_hock']
                        print('Hock params', m, r_hock)
                        if self.CONTAINER in [None,""]:
                            Ipot_dir = join(self.ROOTDIR,f"aux_data/Ipot/Ipot_DOY_{self.RESOLUTION}_{self.BASIN}.nc")
                        else:
                            Ipot_dir = join(self.CONTAINER,f"Ipot_DOY_{self.RESOLUTION}_{self.BASIN}.nc")

                        Ipot = xr.open_dataset(Ipot_dir)
                        M = m + r_hock * Ipot
                        M = M.radiation.rename(dict(x = 'lon',y = 'lat'))

                        # new_forcing = xr.open_dataset("/home/pwiersma/scratch/Data/ewatercycle/wflow_Julia_forcing/wflow_MeteoSwiss_1000m_Dischma_2015_2020.nc")
                        M_forcing = xr.ones_like(new_forcing['pr'])#xr.ones_like(new_forcing)['pr']
                        tttime = M_forcing.time

                        for d in range(1,366):
                            M_forcing.loc[dict(time = tttime[tttime.dt.dayofyear == d])] = M.sel(day_of_year = d)
                        M_forcing = xr.where(M_forcing<m,M_forcing.quantile(0.01),M_forcing)
                        new_forcing['Cfmax'] = M_forcing


                    elif self.MODEL_DICS[name]['DD'] == 'subgrid_seasonal':
                        print('subgrid_seasonal')
                        DOY = new_forcing.time.dt.dayofyear
                        DD_min = self.MODEL_DICS[name]['DD_min']
                        DD_max = self.MODEL_DICS[name]['DD_max']
                        
                        fsca = xr.open_dataset(join("/mnt/scratch_pwiersma/PauWiersma/Data/","SnowCover/fSCA_1km_Jonschwil_Synthetic_Landsat.nc")).rename(
                            dict(longitude = 'x',latitude = 'y'))['__xarray_dataarray_variable__']
                        if 'fsca_zero_to_one' in self.MODEL_DICS[name].keys():
                            fsca = xr.where(fsca==0,1,fsca)
                            
                        DD =  fsca * (((DD_min + DD_max)/2) + ((DD_max-DD_min)/2)*np.sin((2*np.pi*(DOY-80.25)/365.25)))
                    
                        mask = new_forcing.time.isin(DD.time)
                    
                        new_forcing['Cfmax'] = xr.ones_like(new_forcing['pr']) 
                        new_forcing['Cfmax'].loc[dict(time = new_forcing.time[mask])] *= DD
                        new_forcing['Cfmax'].loc[dict(time = new_forcing.time[~mask])] *= default_cfmax
                #Save to file under new name
                #TODO make sure forcing files with same DD are reused 
                # new_forcing_dir = os.path.join(self.MODEL_DICS[name]['infolder'],
                #                         f"{orig_forcing[:-3]}_DD_{name}.nc")
                
                # if 'csnow' in self.MODEL_DICS[name].keys():
                #     #There's already a newly made forcing file from csnow correction, it should be removed 
                #     os.remove(orig_forcing)
                    
                # if "CV" in self.MODEL_DICS[name].keys():
                #     CV = self.MODEL_DICS[name]["CV"]
                #     SWEmax = xr.open_dataset(f"/home/pwiersma/scratch/Data/SLF/OSHD_1km_latlon_{self.BASIN}_yearlymax.nc")['swee_all']
                    
                #     print("Applying fsca reduction to melt factor with CV")
                if 'vegcf' in self.MODEL_DICS[name].keys():
                    print("Correction vegetation melt rate")
                    # forest_fraction = xr.open_dataset(f"/home/pwiersma/scratch/Data/SLF/OSHD/OSHD_{self.RESOLUTION}_latlon_{self.BASIN}.nc")['forest']
                    # if self.OSHD =='OSHD':
                    if self.CONTAINER in [None,'']:
                        root = "/home/pwiersma/scratch/Data/SLF/OSHD"
                    else:
                        root = self.CONTAINER
                    forest_fraction = SnowClass(self.BASIN).load_OSHD(resolution = self.RESOLUTION, root = root)['forest']
                    # elif self.OSHD == 'FSM':
                    #     FSM = SnowClass(self.BASIN).load_FSM(resolution = self.RESOLUTION)

                    forest_fraction = forest_fraction.rename(dict(x ='lon',y = 'lat' ))
                    vegcf = self.MODEL_DICS[name]['vegcf']
                    correction  = vegcf* forest_fraction +1
                    new_forcing['Cfmax'] *= correction  
                    # new_forcing_dir = f"{new_forcing_dir[:-3]}_vegcf_{vegcf}.nc"
                if 'petcf' in self.MODEL_DICS[name].keys():
                    print("Correction petcf")
                    petcf = self.MODEL_DICS[name]['petcf']
                    new_forcing['pet'] *= petcf
                    # new_forcing_dir = f"{new_forcing_dir[:-3]}_petcf_{petcf}.nc"
                if 'petcf_seasonal' in self.MODEL_DICS[name].keys():
                    if self.MODEL_DICS[name]['petcf_seasonal']:
                        print('Correcting pet seasonally')
                        if self.CONTAINER in [None,'']:
                            petcf_seasonal = pd.read_csv(join(self.ROOTDIR,"aux_data/pet_comparison_monthly.csv"),
                                                     index_col = 0)
                        else:
                            petcf_seasonal = pd.read_csv(join(self.CONTAINER,"pet_comparison_monthly.csv"),
                                                     index_col = 0)
                        for m in range(1,13):
                            new_forcing['pet'].loc[new_forcing.time.dt.month ==m] *= petcf_seasonal.loc[self.BASIN,str(m)]
                        # new_forcing_dir = f"{new_forcing_dir[:-3]}_petcf_seasonal.nc"
                if 'soilerrdic' in self.MODEL_DICS[name].keys():
                    paths = self.MODEL_DICS[name]['soilerrdic']
                    if paths !=None:
                        print("Applying soil error")
                        # orig_staticmaps = xr.open_dataset(self.MODEL_DICS[name]['full_staticmaps_path'])
                        # new_staticmaps = orig_staticmaps.copy()
                        for par, rf_path in paths.items():
                            rf = np.loadtxt(rf_path)
                            print(par, np.mean(rf))
                            new_staticmaps[par] *= rf
                            new_staticmaps[par] = new_staticmaps[par].where(new_staticmaps[par]>0,0)
                            # plt.figure()
                            # new_staticmaps[par].plot()
                            # plt.show()

                            # diff = new_staticmaps[par] - orig_staticmaps[par]
                            # plt.figure()
                            # diff.plot()
                            # plt.show()
                        # new_staticmaps_fname = f"{config['input']['path_static'][:-3]}_soilerrdic.nc"
                        # new_staticmaps.to_netcdf(join(infolder,new_staticmaps_fname))
                        # new_staticmaps.close()
                        # config['input']['path_static'] = new_staticmaps_fname
                        # self.remove_sm.append(join(infolder,new_staticmaps_fname))
                        # self.MODEL_DICS[name]['full_staticmaps_path'] = os.path.join(infolder,config['input']['path_static'])
                        print(os.path.join(infolder,config['input']['path_static']))
                    else:
                        print("No synthetic soil error")

                
                if self.MODEL_DICS[name].get('TI') or self.MODEL_DICS[name].get('EB'):
                    C = SnowClass(self.BASIN)
                    C.mask = new_staticmaps['wflow_dem']
                    if self.CONTAINER in [None,'']:
                        root = "/home/pwiersma/scratch/Data/ewatercycle/OSHD"
                    else:
                        root = self.CONTAINER
                    if self.MODEL_DICS[name].get('EB'):
                        print("Applying EB-OSHD")
                        FSM = C.load_FSM(resolution = self.RESOLUTION, root = root)
                        runoff = FSM['romc_all'].rename(dict(x = 'lon',y = 'lat')).sel(time = slice(new_forcing.time[0],new_forcing.time[-1]))
                        runoff = runoff.reindex(time = new_forcing.time, fill_value = 0)
                    if self.MODEL_DICS[name].get('TI'):
                        print("Applying TI-OSHD")
                        OSHD = C.load_OSHD(resolution = self.RESOLUTION,root = root)
                        runoff = OSHD['romc_all'].rename(dict(x = 'lon',y = 'lat')).sel(time = slice(new_forcing.time[0],new_forcing.time[-1]))
                        runoff = runoff.reindex(time = new_forcing.time, fill_value = 0).astype('float32')
                    config['input']['forcing'].append('vertical.extra_melt')
                    config['input']['vertical'].pop('extra_melt')
                    config['input']['vertical'].append('extra_melt','extra_melt')
                    new_forcing['extra_melt'] = new_forcing['pr'] * 0
                    new_forcing['extra_melt'] = runoff



                    
                    #for testing with synthetic obs
                    # output_file = '/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV/data/output_Riale_di_Calneggia_a_False/output_Riale_di_Calneggia_a_False.nc'
                    # C.load_SWE_julia(output_file, downscale=True,start_year = self.START_YEAR,end_year = self.END_YEAR)
                    # runoff = C.SWE.diff(dim = 'time',label = 'lower').rename(dict(latitude = 'lat',longitude = 'lon'))
                    # runoff = xr.where(runoff>0,0,runoff)*-1 * (1/self.MODEL_DICS[name]['rfcf'])
                    # runoff = runoff.sel(time = new_forcing.time)
                    
                    # #Set TTI to zero
                    # new_staticmaps['TTI'] *=0
                    # #Make a mask of all temperatures below TT
                    # # TT= config['input']['vertical']['tt']['offset'] 
                    # TT = 0
                    
                    # mask = new_forcing['tas'] < TT
                    # mask = xr.where(np.isnan(new_forcing['tas']),np.nan,mask)
                    # #Remove all precip in the mask
                    # new_forcing['pr'] = xr.where(mask,0,new_forcing['pr'])
                    
                    # #Set all T to TT in the mask
                    # new_forcing['tas'] = xr.where(mask,TT, new_forcing['tas'])

                    # #make sure rfcf does not apply on the masked areas 


                    # #Add OSHD as rainfall to precip
                    # #Where the indices of runoff and new_forcing overlap, add the runoff to the precip
                    # overlap_indices = new_forcing.indexes['time'].intersection(runoff.indexes['time'])
                    # rainfall_correction = self.MODEL_DICS[name]['rfcf']
                    # new_forcing['pr'].loc[overlap_indices] =  new_forcing['pr'].loc[overlap_indices] + (runoff.loc[overlap_indices]/rainfall_correction)
                    # #Divide by rainfall correction factor, because later you gonna multiply with it again     



                    # new_forcing_dir = f"{new_forcing_dir[:-3]}_OSHD1km.nc"

                
                new_forcing.to_netcdf(new_forcing_fname)
                new_staticmaps.to_netcdf(new_staticmaps_fname)
                #Adjust forcing path in config file
                config['input']['path_forcing'] = new_forcing_fname
                config['input']['path_static'] = new_staticmaps_fname
                #List vertical.cfmax under forcing
                config['input']['forcing'].append('vertical.cfmax')
                #Remove cfmax.value = 2.5 and write cfmax = "Cfmax"
                try:
                    config['input']['vertical']['cfmax'] = 'Cfmax'
                except:
                    config['input']['vertical']['cfmax'] = 'Cfmax'
            # if 'ST_scale' in self.MODEL_DICS[name].keys():
                #Load staticmaps, apply scaling, save to new file, adjust config file
                # print("Applying elevation-dependent melt temperature correction")
                # orig_staticmaps = xr.open_dataset(self.MODEL_DICS[name]['full_staticmaps_path'])
                # ST_scale = self.MODEL_DICS[name]['SoilThickness']
                # ST_zscale = self.MODEL_DICS[name]['ST_zscale']
                # dem = orig_staticmaps['wflow_dem']
                # norm_dem = dem/np.nanmean(dem)
            
                
                # plusmin_dem = xr.where(norm_dem<=1, (1/-norm_dem)+1, norm_dem -1) #0.5 becomes 1, 2 becomes 1

                # if ST_zscale<0:
                #     zscaled_ST = xr.where(plusmind_dem<1, plusmin_dem * ST_zscale * ST_scale )

        
                # scaled_ST = plusmin_dem * ST_scale
                # new_staticmaps = orig_staticmaps.copy()
                # new_staticmaps['TT'] = orig_staticmaps['TT'] + ST_offset + scaled_ST
                # # plt.figure()
                # # new_staticmaps['TT'].plot()
                # new_staticmaps_fname = f"{config['input']['path_static'][:-3]}_tt_scale_{ST_offset}_{ST_scale}.nc"
                # new_staticmaps.to_netcdf(join(infolder,new_staticmaps_fname))
                # new_staticmaps.close()
                # config['input']['path_static'] = str(new_staticmaps_fname)
                # # self.remove_sm.append(join(infolder,new_staticmaps_fname))
                # self.MODEL_DICS[name]['full_staticmaps_path'] = os.path.join(infolder,config['input']['path_static'])
                # try:
                #     config['input']['vertical']['tt']['offset'] = 0
                # except:
                #     config['input']['vertical']['tt']['offset'] = 0




                # print("Applying Topographic index-dependent scaling of SoilThickness")
                # orig_staticmaps = xr.open_dataset(self.MODEL_DICS[name]['full_staticmaps_path'])
                # orig_soilthickness = orig_staticmaps['SoilThickness']
                # ST_scale = self.MODEL_DICS[name]['SoilThickness']
                # ST_zscale = self.MODEL_DICS[name]['ST_zscale']
                # if 'scaled_dem' not in dir(self):
                #     dem = orig_staticmaps['wflow_dem']
                #     scaled_dem = self.scale_dem(dem,below_avg_to_0 = False)
                # else:
                #     scaled_dem = self.scaled_dem

                # #scaled_dem between let's say 0.5 and 2
                # #eventually we want 0.5 to get all the soilthickness reduction and 2 to get none 
                # zscaled = scaled_dem * (ST_zscale-1) +1 
                # #To get low precip low and high precip high elevation: ST = 0.8, ST_scale = 2
                # #to Get high precip low and low precip high elevation: ST = 1.5, ST_scale = 0.7

                
                # new_staticmaps = orig_staticmaps.copy()
                # new_staticmaps['SoilThickness'] =      scaled_ST * config['input']['vertical']['ST']['value']
                # new_staticmaps_fname = f"{config['input']['path_static'][:-3]}_ST_scale_{ST_scale+np.random.uniform(0.1,1)/1000}.nc"
            
                
                # new_staticmaps.to_netcdf(join(infolder,new_staticmaps_fname))
                # new_staticmaps.close()
                # config['input']['path_static'] = new_staticmaps_fname
                # self.remove_sm.append(join(infolder,new_staticmaps_fname))
                # self.MODEL_DICS[name]['full_staticmaps_path'] = os.path.join(infolder,config['input']['path_static'])
                # try:
                #     config['input']['vertical']['ST']  = 'ST'
                # except:
                #     config['input']['vertical']['ST']  = 'ST'



            
            with open(config_out, mode = "wt") as fp:
                tomlkit.dump(config,fp)
            
            shutil.copy(config_out,join(outfolder,f"sbm_config_{self.BASIN}_{name}.toml"))
            
    def scale_dem(self,dem,below_avg_to_0=False):
        # dem = rioxr.open_rasterio(join(model_instance_dir,'staticmaps/wflow_dem.map'),masked=True)
        norm_dem = dem/np.nanmean(dem)
        scaled_dem  = (norm_dem-1)/(np.nanmax(norm_dem)-1)
        #what does this mean?  doesnt make much sense? Only for above_average values, then the max is one in this case (and 2 after processing)
        # and the mean corresponds to 0
        if below_avg_to_0 == True:
            scaled_dem = xr.where(scaled_dem<0,0,scaled_dem) #Everything below the average elevation is removed
        self.scaled_dem = scaled_dem
        return scaled_dem
    def generate_MS_forcing_if_needed(self,forcing_name):

        self.FORCING_FILE = forcing_name

        # forcing_file_orig = join(self.ROOTDIR,"wflow_Julia_forcing",forcing_name)
        if not self.CONTAINER in [None,'']:
            dest = join(self.CONTAINER,forcing_name)
            if os.path.isfile(dest):
                print("Forcing file found in container")
                return 
        else: 
            dest = join(self.PARSETDIR,"data/input",forcing_name)
            if os.path.isfile(dest):
                print("Forcing file found in input folder")
                return
        forcing_file_datafolder = join(self.ROOTDIR,"wflow_Julia_forcing",forcing_name)
        if os.path.isfile(forcing_file_datafolder):
            print(f"Copying file from wflow_Julia_forcing to {dest}")
            shutil.copy(forcing_file_datafolder, dest)
        else:
            from generate_forcing import generate_MS_forcing
            print("Generating forcing is needed")
            generate_MS_forcing(fname = forcing_file_datafolder, 
                                basin = self.BASIN, 
                                resolution =self.RESOLUTION,
                                start_year = self.START_YEAR-1,
                                end_year = self.END_YEAR)
            shutil.copy(forcing_file_datafolder, dest)


        # if self.CONTAINER in [None,'']:
        #     forcing_file_orig = ""
        # else:
        #     forcing_file_orig = join(self.CONTAINER,forcing_name)
        # forcing_file_datafolder = join(self.ROOTDIR,"wflow_Julia_forcing",forcing_name)

        # forcing_file_input = join(self.PARSETDIR,"data/input",forcing_name)
        # if not os.path.isfile(forcing_file_orig):
        #     if os.path.isfile(forcing_file_datafolder):
        #         forcing_file_orig = forcing_file_datafolder
        #     else:
        #         from generate_forcing import generate_MS_forcing
        #         print("Generating forcing is needed")
        #         generate_MS_forcing(fname = forcing_file_datafolder, 
        #                             basin = self.BASIN, 
        #                             resolution =self.RESOLUTION,
        #                             start_year = self.START_YEAR-1,
        #                             end_year = self.END_YEAR)
        # if not os.path.isfile(forcing_file_input):
        #     print("Copying file from wflow_Julia_forcing to input folder")
        #     shutil.copy(forcing_file_orig, forcing_file_input)
        # self.FORCING_FILE = forcing_name
    def check_staticmaps(self,staticmaps_name):
            if self.CONTAINER in [None,'']:
                staticmaps_file_orig = ""
            else:
                staticmaps_file_orig = join(self.CONTAINER,staticmaps_name)
            staticmaps_file_datafolder = join(self.ROOTDIR,"wflow_staticmaps",staticmaps_name)
            staticmaps_file_input = join(self.PARSETDIR,"data/input",staticmaps_name)
            # staticmaps_file_hydromt = join(f"/home/pwiersma/scratch/Data/HydroMT/model_builds/wflow_{self.BASIN}_{self.RESOLUTION}_{hydromt_name}/staticmaps.nc")
            if not os.path.isfile(staticmaps_file_orig):
                if os.path.isfile(staticmaps_file_datafolder):
                    staticmaps_file_orig = staticmaps_file_datafolder
                else:
                    print("Generating staticmaps is needed")
                    return
            if not os.path.isfile(staticmaps_file_input):
                print("Copying file from wflow_staticmaps to input folder")
                shutil.copy(staticmaps_file_orig, staticmaps_file_input)
            self.STATICMAPS_FILE = staticmaps_name
    def create_single_model(self,name):
        print (f"Creating {name} model")

        #PArameter set 
        parset = ParameterSet(name = name, 
                                directory = self.PARSETDIR,
                                target_model = "WflowJl",
                                config = f"sbm_config_{self.BASIN}_{name}.toml")
        


        self.models[name] = WflowJl(parameter_set = parset)
        print("initializing...")
        self.models[name].setup()
        print("setup done")
        self.models[name].initialize(self.MODEL_DICS[name]['config_dir'])
        print(f"Created {name} model")
    def create_models(self):
        self.models = {}
        # self.config_files = {}
        
        if not 'MODEL_DICS' in dir(self):
            print('No alternative parameter sets defined, use standard parameter set')
            self.MODEL_DICS = {'STANDARD':None}
        
        for name in self.MODEL_DICS.keys():
            # print(self.MODEL_DICS[name]['config']['input']['path_static'])
            self.create_single_model(name)
        # if threads is None:
        #     pool = Pool()
        # else:
        #     pool = Pool(nodes=threads)
        # # Run parallel models
        # print("Mapping now")
        # pool.map(
        #     self.create_single_model,
        #     self.MODEL_DICS.keys())
        # pool.close()
        # pool.join()
    
    def standard_run(self, key, test):        
        #settrace(f)
        t0 = time.time()
        # count=0        
        print('Start standard loop') 
        
        END_TIMESTAMP = datetime.strptime(f'01-10-{self.END_YEAR} 00:00', '%d-%m-%Y %H:%M').timestamp()
        START_TIMESTAMP = datetime.strptime(f'01-10-{self.START_YEAR-1} 00:00', '%d-%m-%Y %H:%M').timestamp()
        TIMESTAMP_DIFF = END_TIMESTAMP - START_TIMESTAMP
        if test == True:
            TIMESTAMP_DIFF = 86400.0
        self.models[key].bmi.update_until(TIMESTAMP_DIFF)
        seconds = time.time()-t0 

        print(key,seconds/60,'minutes')
        # print(key,(time.time()-t0)/count,' seconds per step')
        
        return 
    # def degree_day_runs(self,key,test):
        
    #     END_TIMESTAMP = datetime.strptime(f'01-10-{self.END_YEAR} 00:00', '%d-%m-%Y %H:%M').timestamp()
    #     model = self.models[key]
    #     if self.MODEL_DICS[key]['DD'] =='static':
    #         print("Static DD run starts")
    #         time0 = time.time()
    #         model.update_until(END_TIMESTAMP)
    #         seconds = time.time()-time0 
    #         print(f"{seconds/60} minutes have passed. ")
        
    #     elif self.MODEL_DICS[key]['DD'] =='seasonal':
    #         i = 0
    #         while  model.get_current_time()!=END_TIMESTAMP:
    #             model.update()
    #             i+= 1
    #             numpy_time = self.time_as_datetime(model.get_current_time())
    #             DOY = numpy_time.astype(datetime).timetuple().tm_yday
                
    #             DD_min = self.MODEL_DICS[key]['DD_min']
    #             DD_max = self.MODEL_DICS[key]['DD_max']
  
    #             current_cfmax = self.get_value_to_xarray(key,'vertical.cfmax').rename({'lat':'latitude','lon':'longitude'})
    #             new_cfmax = current_cfmax * 0 + (((DD_min + DD_max)/2) + ((DD_max-DD_min)/2)*np.sin((2*np.pi*(DOY-80.25)/365.25)))
 
    #             self.set_xarray_as_value(key,new_cfmax, "vertical.cfmax")
    #     elif self.MODEL_DICS[key]['DD'] == 'subgrid_seasonal':
            
    #         # C= SnowClass()
    #         # # C.load_dem("/home/tesla-k20c/ssd/pau/ewatercycle/aux_data/wflow_dem.map")
    #         # C.mask = self.MODEL_DICS[key]['staticmaps']['wflow_dem'].rename({'lon':'longitude','lat':'latitude'})
    #         # C.load_landsat(join("/home/tesla-k20c/data/pau","SnowMaps/RealLandsatSnowCover"),
    #         #                 reproject_to_mask = False,
    #         #                 transform_to_epsg4326=True,
    #         #                 start_year = self.START_YEAR,
    #         #                 end_year = self.END_YEAR)
    #         # C.load_shadow()
    #         # C.shadow = C.shadow.interp_like(C.landsat) 
            
    #         # masked_landsat = C.mask_shadow(C.landsat) #Double memory
    #         # C.load_simulation_files(os.path.join("/mnt/scratch_pwiersma/PauWiersma/Data/","SnowCover/Synthetic/SSM_2feb23/maps"),
    #         #                         self.START_YEAR,self.END_YEAR)
    #         synthetic_fSCA = xr.open_dataset(join("/mnt/scratch_pwiersma/PauWiersma/Data/","SnowCover/fSCA_1km_Jonschwil.nc"))
    #         landsat_fSCA = xr.open_dataset(join("/mnt/scratch_pwiersma/PauWiersma/Data/","SnowCover/fSCA_1km_Jonschwil_Landsat.nc"))            
    #         i = 0
    #         while  model.get_current_time()!=END_TIMESTAMP:
    #             model.update()
    #             i+= 1
    #             numpy_time = self.time_as_datetime(model.get_current_time())
    #             DOY = numpy_time.astype(datetime).timetuple().tm_yday
    #             # print(numpy_time, DOY)
                
    #             DD_min = self.MODEL_DICS[key]['DD_min']
    #             DD_max = self.MODEL_DICS[key]['DD_max']
    #             print(numpy_time)
    #             if numpy_time in C.simfiles.index:
    #                 print("fsca from synthetic")
    #                 synthetic = C.load_single_sim(numpy_time,reproject_to_mask=False,transform_to_epsg4326=True)
    #                 synthetic = C.mask_shadow(synthetic)
                    
    #                 fsca = C.compute_fSCA(synthetic, C.mask)
             
    #             elif numpy_time in C.landsat.time:
    #                 print(f"fsca from Landsat")
    #                 fsca = C.compute_fSCA(masked_landsat.snowcover.sel({'time':numpy_time}), C.mask)
    #             else:
    #                 print('numpytime in neither landsat or synthetic...')
    #                 continue
    #             # # fsca.plot()
    #             # # plt.show()
                
    #             #When there's no snow cover in SSM, take default melt factor to make it disappear quickly (normally actually)
    #             if i==1:
    #                 print("All fsca==0 are set to 1")
    #             fsca = xr.where(fsca==0,1,fsca)
                
    #             current_cfmax = self.get_value_to_xarray(key,'vertical.cfmax').rename({'lat':'latitude','lon':'longitude'})
    #             seasonal_cfmax = current_cfmax * 0 + (((DD_min + DD_max)/2) + ((DD_max-DD_min)/2)*np.sin((2*np.pi*(DOY-80.25)/365.25)))
    #             # new_cfmax = current_cfmax * 0 + fsca *(((DD_min + DD_max)/2) + ((DD_max-DD_min)/2)*np.sin((2*np.pi*(DOY-80.25)/365.25))) 
    #             new_cfmax = xr.where((~np.isnan(current_cfmax)&(~np.isnan(fsca))),
    #                                  fsca * seasonal_cfmax ,
    #                                  seasonal_cfmax)
                

    #             # print(f"new_cfmax {new_cfmax.mean()}")
    #             # new_cfmax.plot()
    #             # plt.show()
    #             self.set_xarray_as_value(key,new_cfmax, "vertical.cfmax")
    #             # print(model.get_value("vertical.cfmax").mean())
    #             # self.get_value_to_xarray(model,staticmaps.wflow_dem,"vertical.snow").plot()
    #             # plt.title(numpy_time)
    #             # plt.show()
    #     return 
    # @profile
    # def series_runs(self, run_func,test=False):
    #     model_keys = self.MODEL_DICS.keys()
    #     results  = map(run_func,model_keys, [test for i in model_keys])
    #     return results
    
    def series_runs(self, run_func, test=False):
        # Get the model keys
        model_keys = self.models.keys()
    
        # Run the models in parallel using a loop
        # results = []
        for model_key in model_keys:
            run_func(model_key, test)
            # results.append(result)
    
        # Return the results
        return 


    def finalize_runs(self):
        if len(self.remove_sm)>0:
            for sm in self.remove_sm:
                try:
                    os.remove(sm)
                    self.remove_sm = []
                except OSError as e:
                    print(f"Error removing file: {e}")
                    
            
            
            
            
            # for name in self.MODEL_DICS.keys():
            #     staticpath = join(self.MODEL_DICS[name]['infolder'] ,
            #                    self.MODEL_DICS[name]['config']['input']['path_static'])
            #     if os.path.isfile(staticpath):
            #         try:
            #             os.remove(staticpath)
            #         except OSError as e:
            #             print(f"Error removing file: {e}")
        for model in self.models.values():
            model.finalize()
    def remove_forcing(self):
        for key in self.MODEL_DICS.keys():
            if 'csnow' in self.MODEL_DICS[key]:
                os.remove(self.MODEL_DICS[key]['config']['input']['path_forcing'])
            elif 'DD' in self.MODEL_DICS[key]:
                os.remove(self.MODEL_DICS[key]['config']['input']['path_forcing'])
                # forcing = xr.open_dataset(self.MODEL_DICS[key]['config']['input']['path_forcing'])
                
    def load_stations(self,STATION_NAMES):
        self.STATION_NAMES = STATION_NAMES
        self.stations = {}
        self.station_lats = []
        self.station_lons = []
        # self.daterange = pd.date_range(self.forcing.start_time,self.forcing.end_time).tz_localize(None)
        self.daterange = pd.date_range(f"{self.START_YEAR-1}-10-01",f'{self.END_YEAR}-09-30', freq = 'D')
        for name in STATION_NAMES:
            self.stations[name] = SwissStation(name)
            if self.SYNTHETIC_OBS == None:
                if self.CONTAINER in [None,'']:
                    root = join(self.ROOTDIR,'Discharge_data')
                else:
                    root = self.CONTAINER
                # self.stations[name].read_station(self.START_YEAR,self.END_YEAR,join(self.ROOTDIR,'Discharge_data'))
                self.stations[name].read_station(self.START_YEAR,self.END_YEAR,root)
            else:
                print('Gathering synthetic observations in evaluation()')
                # synthetic_dir = join(self.ROOTDIR,"experiments/wflow_julia_1000m_2005_2015_JLLRV/SYNTHETIC_OBS",
                #                  f"{self.SYNTHETIC_OBS}")
                out_df = pd.read_csv(self.SYNTHETIC_OBS,index_col = 0,parse_dates = True)
                cols = out_df.columns
                out_df = out_df.rename(columns = {cols[0]:'obs'})


                if out_df.index[0] != self.daterange[0]:
                    missing_dates = pd.date_range(start=self.daterange[0], end=out_df.index[0] - pd.Timedelta(days=1), freq='D')
                    missing_data = pd.DataFrame({'obs': [0] * len(missing_dates)}, index=missing_dates)
                    out_df = pd.concat([missing_data, out_df])
                out_df = out_df.loc[self.daterange]
                self.stations[name].obs = out_df
                self.stations[name].obs.index.name = 'time'
                self.stations[name].obs.attrs['unit'] = 'm3/s'
            self.station_lats.append(self.stations[name].lat)
            self.station_lons.append(self.stations[name].lon)
            self.stations[name].model_dataframe(self.daterange)

    def stations_combine(self):
        for station in self.stations.values():
            station.combine()

    def standard_plot(self,save_figs = False, test = False,forcing_dir = None,skip_first_year = False,
                      add_snow = False, add_rain = True):
        key0 = list(self.MODEL_DICS.keys())[0]
        #TODO make this work also when I use different forcings
        if forcing_dir == None:
            forcing = xr.open_dataset(os.path.join(self.MODEL_DICS[key0]['infolder'],
                                    self.MODEL_DICS[key0]['config']['input']['path_forcing']))#.drop_vars('spatial_ref')
        else: forcing = xr.open_dataset(forcing_dir)
        pr_mean = forcing['pr'].mean(dim=('lon','lat'))
        pr_df = pr_mean.to_dataframe().rename(columns={'pr':'MeteoSwiss (2km)'})

        if add_snow:
            #add an empty df like pr_df 
            melt_df = pd.DataFrame()
            for key in self.MODEL_DICS.keys():
                file = join(self.MODEL_DICS[key]['outfolder'],
                            self.MODEL_DICS[key]['config']['output']['path'])
                SWE_dry = xr.open_dataset(file).snow
                SWE_wet = xr.open_dataset(file).snowwater
                SWE = SWE_dry + SWE_wet
                SWE_diff = (SWE.diff(dim = 'time')/1000) * self.grid_area *1e6
                SWE_diff = xr.where(SWE_diff>0,0,SWE_diff)
                SWE_melt = SWE_diff.sum(dim=['lat','lon'])
                melt_df[key] = SWE_melt.to_dataframe(name = key)
                
            C = SnowClass(self.BASIN)
            #load OSHD
            OSHD_diff = C.load_OSHD()['swee_all'].diff(dim = 'time')
            OSHD_diff = xr.where(OSHD_diff>0,0,OSHD_diff)* self.grid_area * 1e6 *0.001
            OSHD_diff_df = OSHD_diff.sum(dim=['x','y']).to_dataframe(name = 'OSHD')
            melt_df['OSHD'] = OSHD_diff_df
            # melt_df = SWE_df.diff()
            # melt_df = melt_df.where(melt_df<0,0)
            basin_area = SwissStation(self.STATION_NAMES[0]).area
            # melt_df = ((melt_df /1000) *(basin_area*1e6)) /(60*60*24) #convert to m3/s
            melt_df /= -60*60*24
        else:
            melt_df =None
        if add_rain ==False:
            pr_df = None

        # pr_df.pop('height')
        for i,station in enumerate(self.stations.values()):
            
            for year in range(self.START_YEAR,self.END_YEAR+1):    
                if (skip_first_year ==True) & (year == self.START_YEAR):
                    continue
                fig,axes = hydrograph_lines(
                   discharge=station.combined.loc[slice(f"{year-1}-10",f'{year}-09')],
                   reference='obs',
                   sim_names = list(self.MODEL_DICS.keys()),
                   title=station.name+' Hydrograph'+self.RESOLUTION,
                    precipitation=pr_df,
                    melt = melt_df ,
                   figsize=(15,10),filename = None,grid=True)
                if test==True:
                    break
                if save_figs is not False:
                    plt.savefig(join(save_figs,f"wflow_{self.BASIN}_{self.RESOLUTION}_{self.RUN_NAME}_{str(self.START_YEAR)}_{str(self.END_YEAR)}_{station.name}_plot_{year}.png"),
                                bbox_inches = 'tight',bbox_extra_artists = ax.get_legend())
    def station_FDC(self):
        for station in self.stations.values():
            station.flow_duration_curves()  
    def station_OFs(self,skip_first_year = True):
        for name,station in self.stations.items():
            station.objective_functions(skip_first_year = skip_first_year)
            print(station.of)
    def delete_folders(self):
        print('Deleting all native wflow model folders and copy logfile to log folder')
        for cfg in self.config_files.values():
            src = os.path.join(cfg[1],'run_default','wflow.log')
            dst = os.path.join(self.PARSETDIR,'logs',os.path.basename(cfg[1])+'_wflow.log')
            shutil.move(src,dst)
            shutil.rmtree(cfg[1])
    #TODO
    #def get_value_as_xarray(self, model, )
    def time_as_datetime(self,timestamp):
        return np.datetime64(date.fromtimestamp(timestamp))
    def load_dem(self,name):
        # path = self.MODEL_DICS[name]['config']['input']['path_static']
        self.dem = xr.open_dataset(self.MODEL_DICS[name]['full_staticmaps_path'])['wflow_dem']
        return self.dem
    def get_value_to_xarray(self,name,var_name):
        """mask should be xarray, values should be np.array)"""
        if 'dem' not in dir(self):
            self.dem = self.load_dem(name)
        flat_mask = self.dem.data.flatten()
        mask_indices = (~np.isnan(flat_mask))
        flat_mask[mask_indices] = self.models[name].get_value(var_name)
        values2D = flat_mask.reshape(self.dem.shape)
        values_ds = xr.ones_like(self.dem) * values2D
        values_ds = values_ds.rename(var_name)
        return values_ds
    def set_xarray_as_value(self,name,ds,var_name):
        grid = self.models[name].get_var_grid(var_name)
        shape = self.models[name].get_grid_shape(grid)
        ds_to_flat = ds.data.flatten()
        ds_to_flat = ds_to_flat[~np.isnan(ds_to_flat)]
        if ds_to_flat.size != shape:
            print("Dataset does not match var shape")
            return
        self.models[name].set_value(var_name,ds_to_flat)
        return
    def load_Q(self):
        for key in self.MODEL_DICS.keys():
            netcdf_output = xr.open_dataset(os.path.join(self.MODEL_DICS[key]['outfolder'],
                         self.MODEL_DICS[key]['config']['netcdf']['path'])).sortby('Q_gauges')
            Q = netcdf_output.Q.to_pandas()
            for i,station in enumerate(self.stations.values()):
                station.mod[key] = Q[str(station.number)]
                # station.mod[key] = Q[Q.keys()[i]]
        # for key in self.MODEL_DICS.keys():
        #     csv_file =os.path.join(self.MODEL_DICS[key]['outfolder'],
        #                  self.MODEL_DICS[key]['config']['netcdf']['path'])
    def load_SWE(self):
        SWE_dic = {}
        for key in self.MODEL_DICS.keys():
            with xr.open_dataset(os.path.join(self.MODEL_DICS[key]['outfolder'],
                         self.MODEL_DICS[key]['config']['output']['path'])) as ncfile:
                SWE_dic[key] = ncfile.snow + ncfile.snowwater
            SWE_dic[key] = ncfile.snow + ncfile.snowwater
        return SWE_dic



    def load_forcing(self,key):
        self.MODEL_DICS[key]['forcing'] = xr.open_dataset(os.path.join(self.MODEL_DICS[key]['infolder'],
                        self.MODEL_DICS[key]['config']['input']['path_forcing']))#.drop_vars('spatial_ref')
        return self.MODEL_DICS[key]['forcing']
    
    def save_postrun(self,
                     FOLDER,
                     vars = ['snow','snowwater'],
                     only_Q = False,
                     skip_first_year = True):
        Path(FOLDER).mkdir(parents=True, exist_ok=True)
        for station in self.stations.values():
            if 'of' in dir(station):
                station.of.to_csv(join(FOLDER,f'{station.name}_OFs.csv'))
            Q = station.combined
            if skip_first_year == True:
                Q = Q.loc[slice(f"{self.START_YEAR}-10",f"{self.END_YEAR}-09")]
            Q.to_csv(join(FOLDER,f'{station.name}_Q.csv'))
            
        #scalar output also copied
        for key in self.MODEL_DICS.keys():
            csv =os.path.join(self.MODEL_DICS[key]['outfolder'],
                                                        self.MODEL_DICS[key]['config']['csv']['path'])
            shutil.copy(csv,join(FOLDER,f"{self.BASIN}_{key}.csv"))
        if only_Q ==False:
            if vars != None:
                for var in vars:
                    output_list = []
                    for key in self.MODEL_DICS.keys():
                        output = xr.open_dataset(os.path.join(self.MODEL_DICS[key]['outfolder'],
                                                            self.MODEL_DICS[key]['config']['output']['path']))
                        output_list.append(output[var].rename(f"{var}_{key}"))
                    var_output = xr.merge(output_list)
                    if skip_first_year == True:
                        var_output = var_output.sel(time = slice(f"{self.START_YEAR}-10",f"{self.END_YEAR}-09"))
                    var_output.to_netcdf(join(FOLDER,f"{self.BASIN}_{var}.nc"))

            
    def save_results(self,short_run_name,vars = ['snow','snowwater'],only_Q = False):
        key = list(self.MODEL_DICS.keys())[0]
        joint_folder = os.path.join(self.PARSETDIR,f'data/output_joint_{short_run_name}_{self.START_YEAR}_{self.END_YEAR}')
        if not os.path.isdir(joint_folder):
            os.mkdir(joint_folder)
        base_name = os.path.join(joint_folder,f"wflow_{self.RESOLUTION}_{short_run_name}_{str(self.START_YEAR)}_{str(self.END_YEAR)}_")
        for station in self.stations.values():
            if 'of' in dir(station):
                # station.of.to_csv(base_name+f'{station.name}_OFs.csv')
                station.of.to_csv(base_name+f'OFs.csv')

            # station.combined.to_csv(base_name+f'{station.name}_discharge.csv')
            station.combined.to_csv(base_name+f'discharge.csv')
        
        if only_Q ==False:
            for var in vars:
                output_list = []
                for key in self.MODEL_DICS.keys():
                    output = xr.open_dataset(os.path.join(self.MODEL_DICS[key]['outfolder'],
                                                          self.MODEL_DICS[key]['config']['output']['path']))
                    output_list.append(output[var].rename(f"{var}_{key}"))
                xr.merge(output_list).to_netcdf(base_name+f"{var}.nc")
    def load_state(self, parsetname ):
        fname = f"wflow_state_{self.RESOLUTION}_{self.BASIN}_{self.START_YEAR}_{parsetname}.nc"
        self.instatepath = join(self.ROOTDIR,f'wflow_states/{self.RESOLUTION}',fname)
    def save_state(self, key, parsetname):
        
        fname = f"wflow_state_{self.RESOLUTION}_{self.BASIN}_{self.END_YEAR+1}_{parsetname}.nc"
        state_folder = join(self.ROOTDIR, f"wflow_states/{self.RESOLUTION}")
        orig_path = join(self.MODEL_DICS[key]['outfolder'], self.MODEL_DICS[key]['config']['state']['path_output'])
        shutil.copy(orig_path, join(state_folder,fname))
        
    def csnow(self,name ,csnow, orig_forcing):
        """Deprecated in favor of sfcf native to wflow"""
        staticmaps = xr.open_dataset(self.MODEL_DICS[name]['full_staticmaps_path'])
        dem_withzeros = staticmaps.wflow_dem.fillna(0)

        #Filling all the nans around the catchment, why again? 
        MS_forcing = xr.open_dataset(os.path.join(self.MODEL_DICS[name]['infolder'],orig_forcing))
        MS_forcing['pet'] = MS_forcing['pet'].fillna(MS_forcing['pet'].mean(dim = ['lat','lon']))
        MS_forcing['tas'] = MS_forcing['tas'].fillna(MS_forcing['tas'].mean(dim = 'lon'))

        TTI = staticmaps.TTI.mean()
        TT = staticmaps.TT.mean()

        #partitionaccordig to Verseveld2022, 1 is 100% liquid, 0 is 100% solid
        partition = np.maximum(xr.zeros_like(MS_forcing['tas']),
                                      np.minimum(np.ones_like(MS_forcing['tas']),
                                            (MS_forcing['tas']-TT-0.5*TTI)/TTI))
        snowfall_correction = np.absolute(1-partition) * (csnow-1) +1
        print(f"Correcting MS pr, setting = {name}")
        MS_forcing['pr'] *= snowfall_correction


        MS_forcing = MS_forcing.interp_like(dem_withzeros,kwargs = dict(fill_value = 'extrapolate'))
        MS_forcing.coords['mask'] = xr.where(staticmaps.wflow_subcatch==1,1,0)

        # print('wflow_dem has ', np.count_nonzero(np.isnan(staticmaps.wflow_dem.data)).compute(),' nans')
        # print('MS_Frocing has ', np.count_nonzero(np.isnan(MS_forcing.pr[0].data)),' nans')


        for var in ['pr','tas','pet']:
            MS_forcing[var] = xr.where(dem_withzeros ==0, np.nan, MS_forcing[var])
        # test = MS_forcing[['time','y','x','precip','temp','pet']]
        MS_forcing  = MS_forcing.drop_dims('bnds').rename({'lat':'y','lon':'x'})#,'pr':'precip','tas':'temp'})
        MS_forcing['time'] = MS_forcing.time - np.timedelta64(12,'h')
        MS_forcing['time'] = MS_forcing.time - np.timedelta64(30,'m')
        MS_forcing = MS_forcing.transpose('time','y','x')
        MS_forcing.coords['height'] = 2.0
        MS_forcing['height'].attrs['unit'] = 'm'
        MS_forcing['height'].attrs['standard_name'] = 'height'
        MS_forcing['height'].attrs['long_name'] = 'height'
        MS_forcing['height'].attrs['positive'] = 'up'

        forcing_dir = os.path.join(self.MODEL_DICS[name]['infolder'],f"{orig_forcing[:-3]}_csnow_{csnow}.nc")
        MS_forcing.to_netcdf(forcing_dir)
        return forcing_dir 
              # def prepare_streamflowgrid_output(self):
              #     self.streamflow_grid = {}
              #     for key in self.models.keys():
              #         self.streamflow_grid[key] = []
              # def prepare_snowgrid_output(self):
              #     self.snow_grid = {}
              #     for key in self.models.keys():
                      # self.snow_grid[key] = []
              # def prepare_runoff_output(self):
              #     for key in self.models.keys():
              #         for station in self.stations.values():
              #             station.mod[key] = np.nan 
    # def parallel_runs(self, run_func, threads,test=False):
    #     # Set number of threads (cores) used for parallel run and map threads
    #     #settrace(f)
    #     if threads is None:
    #         pool = Pool()
    #     else:
    #         pool = Pool(nodes=threads)
    #     # Run parallel models
    #     results = pool.map(
    #         run_func,
    #         self.models.keys(),
    #         [test for i in self.models]) #Needs to be iterable 
    #     # self.results = results
    #     # pool.close()
    #     # pool.join()
    #     # snow_grid_list = []
        
    #     #TODO this part should be independent of whether you run parallel or not 
    #     # for result in results:
    #     #     key = list(result.keys())[0]
    #     #     for station_name, station in self.stations.items():
    #     #         station.mod[key].loc[result[key]['time']] = result[key][station_name]
    #     #     # if 'snow_grid' in result[key].keys():
    #     #     #     snow_grid_list.append(result[key]['snow_grid'].rename(f"snow_{key}")) 
    #     #     # snow_grid_dict[k] = result[k]['snow_grid']

    def set_cprec(self,model_instance_dir,parameter_value):
        # forcing_file = xr.open_dataset(self.forcing_dir )
        local_forcing_dir = join(model_instance_dir,self.forcing.netcdfinput)
        forcing_file = xr.open_dataset(local_forcing_dir)
        if 'normalized_dem' not in dir(self):
            scaled_dem = self.scale_dem(model_instance_dir,below_avg_to_0 = False)
        else:
            scaled_dem = self.normalized_dem
            
        cprec_grid = scaled_dem * (parameter_value-1) +1
        cprec_grid = cprec_grid.rename({'x':'lon','y':'lat'})
        # new_forcing = forcing_file.pr*cprec_grid
        # with xr.open_dataset(forcing_dir) as fds:
        #     fds['pr'] = (['time','lat','lon'],fds['pr'].data*cprec_grid.data)
        #     fds.to_netcdf(forcing_dir,mode='a')
        forcing_file['pr'] = (['time','lat','lon'],forcing_file['pr'].data*cprec_grid.data)
        # print(self.forcing_dir)
        # os.remove(self.forcing_dir)
        # forcing_file.to_netcdf(self.forcing_dir)
        print(local_forcing_dir)
        os.remove(local_forcing_dir)
        forcing_file.to_netcdf(local_forcing_dir)
        # plt.figure()
        # cprec_grid.plot()
        # plt.show()



# %%
