#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:28:20 2022

@author: tesla-k20c
"""

# import ewatercycle
# import ewatercycle.parameter_sets
# import ewatercycle.analysis
# ewatercycle.CFG.load_from_file("/home/pwiersma/scratch/Data/ewatercycle/ewatercycle.yaml")

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
# import ewatercycle.forcing
# import ewatercycle.models
import xarray as xr
from scipy import interpolate
import matplotlib.pyplot as plt
import time
from datetime import date
import HydroErr as he



class SwissStation:
    def __init__(self,station_name):
        """ 
        Class for Swiss hydrological stations 
        Input arguments:
            station_name
            station_number (4 digits)
            station_lat
            station_lon
        """
        station_numbers = {'Andelfingen':2044,
                            'Jonschwil':2303,
                          'Mogelsberg':2374,
                          'Rietholz':2414,
                          'Aachbach':7401,
                          'Rindalbach':7102,
                          'Rickenbach':6302,
                          'Luteren':6101,
                          'Iltishag':5902,
                          'Wissthur':5901,
                          'Chlostobel':5804,
                          'Landquart':2150,
                          'Landwasser':2355,
                          'Dischma':2327,
                          'Rom':2617,
                          'Verzasca':2605,
                          'Riale_di_Pincascia':2612,
                           'Ova_da_Cluozza': 2319,
                            'Sitter': 2112,
                            'Kleine_Emme': 2487,
                            'Werthenstein': 2487,
                            'Sense': 2179,
                            'Ilfis': 2603,
                            'Eggiwil': 2409,
                            'Chamuerabach': 2263,
                            'Veveyse': 2486,
                            'Minster': 2300,
                            'Ova_dal_Fuorn': 2304,
                            'Alp': 2609,
                            'Biber': 2604,
                            'Muota': 2499,
                            'Riale_di_Calneggia': 2356,
                            'Chli_Schliere': 2436,
                            'Allenbach': 2232
                          }
        #station_coords slightly shifted to match the model routing
        station_coords = {'Andelfingen':(8.68197964752686069, 47.5965230130429191),
                            'Jonschwil':(9.07947287775765521,47.41267074570048601),
                          'Mogelsberg':(9.123433885445603,47.36418913091697),
                          'Rietholz':(9.012283195812385,47.37607396626375),
                          'Iltishag':(9.233823,47.191392),
                          'Aachbach':(9.122658,47.365205),
                          'Rindalbach':(9.080596,47.409890),
                         'Rickenbach':(9.094107,47.287864),
                             'Luteren':(9.193050,47.225635),
                                 'Wissthur':(9.219441,47.202261),
                                     'Chlostobel':(9.301248,47.194892),
                         'Landquart':(9.6121055,46.9746492),
                         'Landwasser':(9.7899195,46.7578445),
                         'Dischma':(9.8773145,46.7754587),
                         'Rom':(10.45316,46.6296925),
                         'Verzasca':(8.8446202,46.2490202),
                         'Riale_di_Pincascia':(8.8399714,46.2583539),
                             'Ova_da_Cluozza': (10.1183424, 46.6931969),
                            'Sitter': (9.4105757, 47.3319547),
                            'Kleine_Emme': (8.068526, 47.0349015),
                            'Werthenstein': (8.0685262, 47.0349013),
                            'Sense': (7.3512358, 46.8884599),
                            'Ilfis': (7.797701, 46.9379833),
                            'Eggiwil': (7.8046807, 46.8712698),
                            'Chamuerabach': (9.9359513, 46.5694839),
                            'Veveyse': (6.8478776, 46.4687785),
                            'Minster': (8.8137822, 47.0805529),
                            'Ova_dal_Fuorn': (10.1900294, 46.6550163),
                            'Alp': (8.7392767, 47.1509337),
                            'Biber': (8.7207141, 47.1533282),
                            'Muota': (8.7844292, 46.9731421),
                            'Riale_di_Calneggia': (8.5430518, 46.3695832),
                            'Chli_Schliere': (8.2767707, 46.9442437),
                            'Allenbach': (7.5518519, 46.4858821)
                        }
        station_upstream_area = {'Andelfingen':1702,
                            'Jonschwil':493,
                          'Mogelsberg':88.1,
                          'Rietholz':3.19,
                          'Aachbach':19.77,
                          'Rindalbach':7.6,
                          'Rickenbach':15.9,
                          'Luteren':28.9,
                          'Iltishag':84,
                          'Wissthur':17.3,
                          'Chlostobel':39.3,
                          'Landquart':613.7,
                          'Landwasser':186,
                          'Dischma':43,
                          'Rom':128,
                          'Verzasca':185,
                          'Riale_di_Pincascia':44,
                          'Ova_da_Cluozza': 27,
                            'Sitter': 75,
                            'Kleine_Emme': 478,
                            'Werthenstein': 311,
                            'Sense': 351,
                            'Ilfis': 187,
                            'Eggiwil': 124,
                            'Chamuerabach': 73,
                            'Veveyse': 65,
                            'Minster': 59,
                            'Ova_dal_Fuorn': 55,
                            'Alp': 47,
                            'Biber': 32,
                            'Muota': 31,
                            'Riale_di_Calneggia': 24,
                            'Chli_Schliere': 22,
                            'Allenbach': 29}
        self.name = station_name
        self.number = station_numbers[station_name]
        self.lon = station_coords[station_name][0]
        self.lat = station_coords[station_name][1]
        self.area = station_upstream_area[station_name]
        # self.mod = {}
        # self.mod_time = []
    def printfunc(self):
        print(self.name,self.number)
        print('Coords:',(self.lat,self.lon))
    def read_station(self,startyear,endyear,discharge_dir):
        """Loads CH hydrodata based on station number into pd.DataFrame"""
        PATH = glob.glob(join(discharge_dir,'*'+str(self.number)+"*"))[0]
        if 'Export Daten' in PATH: # for St Gallen data
            self.obs = pd.read_csv(PATH,delimiter = ';',index_col = 'Datum',parse_dates = True, usecols = ['Datum','Wert [Kubikmeter pro Sekunde]']).astype(float)
            self.obs = self.obs.rename(columns = {'Wert [Kubikmeter pro Sekunde]':'obs'})
            # self.obs = self.obs.drop(['Datum/Uhrzeit','Zeit'],axis=1)
        elif 'Abfluss_Tagesmittel' in PATH:
            self.obs=pd.read_csv(PATH,delimiter=';',skiprows=8,encoding='cp858',
                                       parse_dates=True,index_col='Zeitstempel',usecols=['Zeitstempel','Wert']).astype(float)
            self.obs = self.obs.rename(columns={'Wert':'obs'})
        self.obs= self.obs[slice(f"{startyear-1}-10",f'{endyear}-09')]
        self.obs.index.name = 'time'
        self.obs.attrs['unit'] = 'm3/s'
        if self.name =='Rietholz':
            self.obs /= 1000
    def model_dataframe(self,pd_daterange):
        self.mod = pd.DataFrame(index=pd_daterange)
    def combine(self):
        if not 'mod' in dir(self):
            print('No modelled time series available yet')
            return
        elif not 'obs' in dir(self):
            print('No observed time series available yet')
            return
        self.combined = self.obs.join(self.mod,how='outer') #make sure both have the same nans? 
    def flow_duration_curves(self,plot=True):
        """ Flow duration curve calculation"""
        self.combined_fd = pd.DataFrame(columns = self.combined.keys(),index = np.linspace(0,100,101))
        for key in self.combined.keys():
            y = self.combined[key].sort_values(ascending=False).values
            x= np.cumsum(y)/np.nansum(y) *100
            xx = self.combined_fd.index
            f = interpolate.interp1d(x,y,fill_value = 'extrapolate')
            yy = f(xx)
            self.combined_fd[key] = yy
        if plot == True:
            plt.figure()
            # plt.plot(self.combined_fd)
            self.combined_fd.plot(legend=True)
            plt.grid(alpha=0.5)
            plt.ylabel('Discharge [m3/s]')
            plt.xlabel('Exceedance probability [-]')
            plt.title(self.name+' Flow-Duration curve')
            plt.semilogy()
            # plt.legend()
    def objective_functions(self,flow_duration = False,months = 'all',skip_first_year = True):
        if flow_duration ==True:
            inner_join = self.combined_fd
        else:
            inner_idx = self.obs.join(self.mod,how='inner').index  
            inner_join = self.combined.loc[inner_idx]
        if months != 'all':
            inner_join = inner_join[inner_join.index.month.isin(months)] 
        if skip_first_year == True:
            years = np.unique(inner_join.index.year)
            inner_join= inner_join[slice(f"{years[1]}-10-01",f"{years[-1]}")]
        kge_values = []
        nse_values = []
        rmse_values = []
        dr_values = []
        relbias_values = []
        for column in self.mod:
            kge_values.append(he.kge_2009(inner_join[column].values, inner_join['obs'].values))
            nse_values.append(he.nse(inner_join[column].values, inner_join['obs'].values))
            rmse_values.append(he.rmse(inner_join[column].values, inner_join['obs'].values))
            dr_values.append(he.dr(inner_join[column].values, inner_join['obs'].values))
            relbias_values.append((inner_join[column].values.sum()-inner_join['obs'].values.sum())/inner_join['obs'].values.sum())
        objective_functions = pd.DataFrame(index = self.mod.keys())
        objective_functions['kge']=kge_values      
        objective_functions['nse']=nse_values
        objective_functions['rmse']=rmse_values
        objective_functions['dr']=dr_values
        objective_functions['relbias'] = relbias_values
        
        self.of = objective_functions
    def load_hydrographs(self,file_path):
        self.combined = pd.read_csv(file_path,index_col = 0,parse_dates = True)
