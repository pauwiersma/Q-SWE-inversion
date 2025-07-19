#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:10:15 2022

@author: tesla-k20c
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import glob
from os.path import join

folder = "/home/tesla-k20c/data/pau/MeteoSuisse/Snow/"
all_files = glob.glob(join(folder,'*hto*d0*data.txt'))

names = {'RIC':'RICKEN',
         'MGB':'MOGELSBERG',
         'RHO':'RIETHOLZ',
         'SPZ':'ST_PETERZELL',
         'WHA':'WILDHAUS',
         'SAE':'SAENTIS',
         'SWA':'SCHWAEGALP1',
         'SLF3SW':'SCHWAEGALP2'}

station_coordinates = {'RIC':[9.059697 ,47.262617],
                       'MGB':[9.137958 ,47.361997],
                       'RHO':[9.01228,47.37608],
                       'SPZ':[9.174725,47.317064],
                       'WHA':[9.367372,47.200889],
                       'SAE':[9.343469,47.249447],
                       'SWA':[9.317192,47.256442],
                       'SLF3SW':[9.317192,47.256442]}
station_elevation = {'RIC':836,
                       'MGB':780,
                       'RHO':715,
                       'SPZ':700,
                       'WHA':998,
                       'SAE':2501,
                       'SWA':1348,
                       'SLF3SW':1348}

files = {}
station_dfs = {}
for name in names.keys():
    files[name] = []
    for file in all_files:
        if name in file:
            files[name].append(file)
            
    # sae_files = files['SAE']
    df_list = []
    for f in files[name]:
        df = pd.read_csv(f,skiprows = 2,
                           delimiter = ';',
                           index_col = 'time',
                           parse_dates = True,
                           na_values = '-').rename(columns = {'stn':'Station',
                                               'hto000d0':'HS',
                                               'qhto000d0':'PI',
                                               'mhto000d0':'MI'})
        df_list.append(df)
    station_df = pd.concat(df_list)                                                      
                                                          
    station_df.attrs['unit'] = 'cm'   
    station_df.attrs['full_name'] = names['SAE']  
    station_df.attrs['coordinates'] = station_coordinates['SAE']
    station_df.attrs['elevation'] = station_elevation['SAE'] 

    station_dfs[name] = station_df
    
for name in names.keys():
    plt.figure()
    station_dfs[name].HS.plot(title=name)
                                                
for name in names.keys():
    station_dfs[name].to_csv(join(folder,f"MeteoSwiss_HS_{names[name]}.csv"))

