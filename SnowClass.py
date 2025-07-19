# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:27:10 2021

Analyse wflow snow cover maps 

@author: -



TODO:
    comparison with meteosuisse maps 
    error maps 
    
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import glob
import datetime as dt
import rioxarray as rioxr
from functools import reduce
from rasterio.enums import Resampling

# import memory_profiler

#%% Load wflow SWE maps and mask to Jonschwil

class SnowClass:
    
    def __init__(self,station_name = 'Jonschwil'):
        # station_coords = {'Andelfingen':(8.68197964752686069, 9.07747287775765521),
        #                     'Jonschwil':(9.07947287775765521,47.41267074570048601),
        #                   'Mogelsberg':(9.123433885445603,47.36418913091697),
        #                   'Rietholz':(9.012283195812385,47.37607396626375)}
        # station_upstream_area = {'Andelfingen':1702,
        #                     'Jonschwil':493,
        #                   'Mogelsberg':88.1,
        #                   'Rietholz':3.19}
        self.basin = station_name 
        # self.basin_area = station_upstream_area[station_name]
        # basins_stations = dict( Jonschwil =  ['MGB','SPZ','WHA','SLF3SW','SLFAM2',
        #                                                         '3SW0','3UI0'],
        #                 Landquart =  ['LAN','KUB','SLFKL2','SLFPAR','SLFKL3',
        #                                                           '5VZ0','5PU0','5SA0','5KU0','5KK0','5KR0'],
        #                 Landwasser =  ['DMA','SLFDA3','SLFWFJ','SLFSLF','SLFDA2',
        #                                                           '5WJ0','5DF0','5DO0','5MA0'],
        #                 Rom     =  ['MUS','BUF','SLFOF2',
        #                                                       '7ST0'],
        #                 Verzasca =  ['SON','SLFFRA'])
        
        #Without SWE from manual stations  that I filled myself
        # basins_stations = dict( Jonschwil =  ['MGB','SPZ','WHA','SLF3SW','SLFAM2'],
        #                 Landquart =  ['LAN','SLFKL2','SLFPAR','SLFKL3','5KU0','5KK0'],
        #                 Landwasser =  ['DMA','SLFDA3','SLFDA2',
        #                                                          '5WJ0','5DF0','5MA0'],
        #                 Rom     =  ['MUS','BUF','SLFOF2', '7ST0'],
        #                 Verzasca =  ['SON','SLFFRA'])
        
        #With SWE from manual stations  that I filled myself
        basins_stations = dict( Jonschwil =  ['MGB','SPZ','WHA','SLF3SW','SLFAM2','3UI0'],
                        Landquart =  ['LAN','SLFKL2','SLFPAR','SLFKL3','5KU0','5KK0',
                                      '5KR0','5SA0','5VZ0','5PU0',],
                        Landwasser =  ['SLFDA3','SLFDA2',    '5WJ0','5DF0','5MA0',
                                                                 '5DO0'], #'DMA',
                        Rom     =  ['MUS','BUF','SLFOF2', '7ST0'],
                        Verzasca =  ['SON','SLFFRA'],
                        Dischma =  ['DMA','5WJ0','5DF0'],#,'SLFFLU'],
                        Riale_di_Calneggia = ['6BG0','SLFBO2','6RO0'])
        try:
            self.station_short_names = basins_stations[station_name]
        except: 
            self.station_short_names = []
        self.SWE_stations = ['3UI0','3SW0','5PU0','5SA0','5KK0','5WJ0','5DF0','5DO0','5MA0','7ST0']
        
        self.long_names = {'RIC':'RICKEN',
                 'MGB':'MOGELSBERG',
                 'RHO':'RIETHOLZ',
                 'SPZ':'ST_PETERZELL',
                 'WHA':'WILDHAUS',
                 'SAE':'SAENTIS',
                 'SWA':'SCHWAEGALP1',
                 'SLF3SW':'SCHWAEGALP2',
                 'LAN':'Landquart',
                 'KUB':'Kublis',
                 'SLFKL2':'Madrisa',
                 'SLFPAR':'Kreuzweg',
                 'SLFKL3':'Gatschiefer',
                 'DMA':'Dischma',
                 'SLFDA3':'Hanengretji',
                 'DAV4':'Frauentobel',
                 'SLFWFJ':'Weissfluhjoch',
                 'SLFSLF':'Stilli',
                 'SLFFLU':'Fluelapass',
                 'SLFDA2':'Barentalli',
                 'MUS': 'Mustair',
                 'BUF':'Buffarola',
                 'SLFOF2':'Murtarol',
                 'SON':'Sonogno',
                 'SLFDTR':'Costa',
                 'SLFFRA':'Efra',
                 'SLFAM2':'Barenfall',
                 'SLFBO2':'Hendar Furggu',
                 '3UI0':'Unterwasser Ilios',
                 '5VZ0':'Valzeina',
                 '5PU0':'Pusserein',
                 '5SA0':'St. Antonien',
                 '5KU0':'Kueblis',
                 '5KR0':'Klosters RhB',
                 '5KK0':'Klosters KW',
                 '5WJ0':'Weissfluhjoch',
                 '5DF0':'Davos Fluelastr.',
                 '5DO0':'Davos WRC Obs',
                 '5MA0':'Matta Frauenkirch',
                 '7ST0':'Sta. Maria',
                 '6BG0':'Bosco Gurin',
                 '6RO0':'Robiei'
                 }
    
        self.station_coordinates = {'RIC':[9.059697 ,47.262617],
                               'MGB':[9.137958 ,47.361997],
                               'RHO':[9.01228,47.37608],
                               'SPZ':[9.174725,47.317064],
                               'WHA':[9.367372,47.200889],
                               'SAE':[9.343469,47.249447],
                               'SWA':[9.317192,47.256442],
                               'SLF3SW':[9.317192,47.256442],
                               'LAN':[9.576428 ,46.964519],
                               'KUB':[9.777506,46.912275],
                               'SLFKL2':[9.8719,46.9097],
                               'SLFPAR':[9.80483,46.85172],
                               'SLFKL3':[9.93167,46.84128],
                               'DMA':[9.882,46.772],
                               'SLFDA3':[9.77410,46.7886],
                               'DAV4':[9.7846,46.784],
                               'SLFWFJ':[9.80888,46.82946],
                               '5WJ0':[9.80888,46.82946],
                               'SLFSLF':[9.8483,46.8129],
                               '5DF0':[9.8483,46.8129],
                               'SLFFLU':[9.9464,46.7523],
                               'SLFDA2':[9.8196,46.6987],
                               'MUS':[10.458519,46.635419],
                               'BUF':[10.267,46.648],
                               'SLFOF2':[10.2886,46.6318],
                               'SON':[8.7883,46.34958],
                               'SLFDTR':[8.8179,46.3521],
                               'SLFFRA':[8.8526,46.3386],
                               'SLFAM2':[9.1468,47.1707],
                               'SLFBO2':[8.4736932,46.3338027],
                               '6BG0':[8.4970693,46.3188851],
                               '6RO0':[8.5130147,46.442437]}
        self.station_elevation = {'RIC':836,
                               'MGB':780,
                               'RHO':715,
                               'SPZ':700,
                               'WHA':998,
                               'SAE':2501,
                               'SWA':1348,
                               'SLF3SW':1348,
                               'LAN':537,
                               'KUB':839,
                               'SLFKL2':2147,
                               'SLFPAR':2290,
                               'SLFKL3':2299,
                               'DMA':1710,
                               'SLFDA3':2455,
                               'DAV4':2330,
                               'SLFWFJ':2536,
                               'SLFSLF':1563,
                               'SLFFLU':2394,
                               'SLFDA2':2558,
                               'MUS': 1248,
                               'BUF':1970,
                               'SLFOF2':2359,
                               'SON':910,
                               'SLFDTR':2170,
                               'SLFFRA':2100,
                               'SLFAM2':1610,
                               'SLFBO2': 2299,
                               '3UI0':1340,
                               '5VZ0':1090,
                               '5PU0':942,
                               '5SA0':1510,
                               '5KU0':815,
                               '5KR0':1188,
                               '5KK0':1190,
                               '5WJ0':2536,
                               '5DF0':1560,
                               '5DO0':1590,
                               '5MA0':1655,
                               '7ST0':1387,
                               '6BG0':1525,
                               '6RO0':1890}
        self.jonas2009_region = {'RIC':3,
                               'MGB':3,
                               'RHO':3,
                               'SPZ':3,
                               'WHA':3,
                               'SAE':3,
                               'SWA':3,
                               'SLF3SW':3,
                               'LAN':5,
                               'KUB':5,
                               'SLFKL2':5,
                               'SLFPAR':5,
                               'SLFKL3':5,
                               'DMA':5,
                               'SLFDA3':5,
                               'DAV4':5,
                               'SLFWFJ':5,
                               'SLFSLF':5,
                               'SLFFLU':5,
                               'SLFDA2':5,
                               'MUS': 7,
                               'SLFOF2':7,
                               'SON':6,
                               'SLFDTR':6,
                               'SLFFRA':6,
                               'SLFAM2':3,
                               'BUF':7,
                               'SLFBO2': 6}
        self.slf_short_names = {  'SLF3SW': 'SLF.3SW',
                                    'WHA': 'MCH.WHA2',
                                    'MGB': 'MCH.MGB2',
                                    'SPZ': 'MCH.SPZ2',
                                    'BUF': 'MCH.BUF2',
                                    'DMA': 'MCH.DMA2',
                                    'LAN': 'MCH.LAN2',
                                    'MUS': 'MCH.MUS2',
                                    'SON': 'MCH.SON2',
                                    '5DF0': 'SLF.5DF',
                                    '5KK0': 'SLF.5KK',
                                    '5KU0': 'SLF.5KU',
                                    '5LQ0': 'SLF.5LQ',
                                    '5MA0': 'SLF.5MA',
                                    '5WJ0': 'SLF.5WJ',
                                    '7BU0': 'SLF.7BU',
                                    '7ST0': 'SLF.7ST',
                                    'SLFBO2': 'SLF.BOG2',
                                    'SLFAM2': 'SLF.AMD2',
                                    'SLFDA2': 'SLF.DAV2',
                                    'SLFDA3': 'SLF.DAV3',
                                    'SLFFRA': 'SLF.FRA2',
                                    'SLFKL2': 'SLF.KLO2',
                                    'SLFKL3': 'SLF.KLO3',
                                    'SLFMTR': 'SLF.MTR2',
                                    'SLFOF2': 'SLF.OFE2',
                                    'SLFPAR': 'SLF.PAR2',
                                    '6BG0':'SLF.6BG',
                                    '6RO0':'SLF.6RO'
                                }                                                   


    def load_landsat(self,folder,start_year=0,end_year=1e6,
                     reproject_to_mask = True,
                     transform_to_epsg4326=False,transform_resolution = 0.0002777777):
        files = glob.glob(join(folder,'*'))

        landsat_list = []
        ls_datelist = []
        # use_file = [False] * len(files)
        use_file = []
        for i,file in enumerate(files):
            # tif = rioxr.open_rasterio(file)
            lsdate = self.get_landsat_date(file)
            if ((lsdate.month>9)|(lsdate.month<7))&((lsdate.year>=start_year)&(lsdate.year<=end_year)):
                # tif = tif.assign_attrs({'time':lsdate})
                # tif['time'] = lsdate
                # landsat_list.append(tif)
                ls_datelist.append(lsdate)
                use_file.append(i)
        
        time_idx = xr.DataArray(pd.Index(ls_datelist),dims = 'time')
        landsat_da = xr.open_mfdataset([files[i] for i in use_file], 
                                       concat_dim = xr.DataArray(time_idx), 
                                       combine='nested'
                                       ).rename({'band_data':'snowcover'}
                                                ).squeeze(dim='band',drop=True,
                                                          ).compute()       
        
        # landsat_da = xr.concat(landsat_list,dim='time').squeeze(
        #     dim='band',drop=True).sortby('time').to_dataset(name='snowcover')
        # landsat_da['ls_bool']=(('time'),np.ones(len(landsat_da.time),dtype='bool'))
        # if (start_year!=None)&(end_year!=None):
        #     landsat_da.sel({'time':slice(start_year,end_year)})

        if reproject_to_mask ==True:
            landsat_da = self.match_mask(landsat_da)
            # landsat_da = landsat_da.rio.reproject_match(self.mask)
            # landsat_da = landsat_da.rename({'x':'longitude','y':'latitude'})
        elif transform_to_epsg4326==True:
            landsat_da = self.match_crs(landsat_da,transform_resolution)
            # landsat_da = landsat_da.rio.reproject(dst_crs="EPSG:4326",resolution = resolution)# 0.001666666)
            # landsat_da = landsat_da.rename({'x':'longitude','y':'latitude'})        
        self.landsat = landsat_da
        return  self.landsat      
    def get_landsat_date(self,Landsat_file):
        basename= os.path.basename(Landsat_file).split('.')[0]
        datestring = basename.split('_')[3]
        datetime    = dt.datetime.strptime(datestring,'%Y%m%d')

        return datetime
    # def check_year(self,file):
    #     if self.get_landsat_date(file).year==year:
    #         return True
    #     else:
    #         return False
    def load_mask(self,folder):
        mask = rioxr.open_rasterio(join(folder,f"{self.basin}_Mask.tif"))
        mask = mask.rename({'x':'longitude','y':'latitude'}).squeeze(dim='band',drop=True)
        self.mask = mask.sortby('longitude',ascending=True).sortby('longitude',ascending=True)
        # self.resampled_mask = mask.interp_like(self.dem,method='nearest')
        # return self.mask
        # mask = )
    def load_dem(self,file):
        m = rioxr.open_rasterio(file,masked=True).rename(
            {'x':'longitude','y':'latitude'}).squeeze(
                dim='band',drop=True).sortby('longitude',ascending=True)
        m.attrs['crs']='epsg:4326'
        self.mask = m
    def load_SWE(self,file,start_year = None,end_year =None):
        SWE = xr.open_dataset(file).interp_like(self.mask)
        # m = self.mask
        # b = m.interp_like(SWE,method='nearest')
        # self.resampled_mask = b
        
        SWE = SWE.where(~np.isnan(self.mask),drop=True)
        if (start_year!=None)&(end_year!=None):
            SWE.sel({'time':slice(f"{start_year-1}-10",f'{end_year}-09')})
        self.SWE = SWE
        # return self.SWE
    def load_SWE_julia(self,file,start_year = None, end_year = None,downscale = False):
        """Load SWE output from Julia, both dry snow and liquid water content in the snow"""
        SWE_dry = xr.open_dataset(file).snow.rename({'lat':'latitude','lon':'longitude'})
        SWE_wet = xr.open_dataset(file).snowwater.rename({'lat':'latitude','lon':'longitude'})
        SWE = SWE_dry + SWE_wet
        if downscale ==True: 
            SWE = SWE.interp_like(self.mask,method = 'nearest',kwargs = dict(fill_value = 'extrapolate'))
        #TODO Mask? 
        if (start_year!=None)&(end_year!=None):
            SWE.sel({'time':slice(f"{start_year-1}-10",f'{end_year}-09')})
        self.SWE = SWE
        self.SWE_dry = SWE_dry
        self.SWE_wet = SWE_wet 
            
    def SWE_to_snowcover(self,threshold=5):
        if not 'SWE' in dir(self):
            print('Calculate SWE first dude')
        self.snowcover = xr.where(self.SWE>threshold,1,0).where(~np.isnan(self.SWE),np.nan)
        return self.snowcover
    def get_simulation_date(self,file):
        basename= os.path.basename(file).split('.')[0]
        datetime    = dt.datetime.strptime(basename,'%Y%m%d')
        return datetime
    def check_sim_year(self,file,start_year,end_year,months):
        date = self.get_simulation_date(file)
        if (date.year>=start_year)&(date.year<=end_year)&(
                np.isin(date.month,months)):
            return True
        else:
            return False
    def load_simulation_files(self,folder,start_year=None,end_year=None,months=[1,2,3,4,5,6,10,11,12]):
        files = glob.glob(join(folder,"*"))
        dates = list(map(self.get_simulation_date,files))
        if (start_year==None)&(end_year==None):
            start_year = dates[0].year
            end_year = dates[-1].year
        boolean = list(map(lambda file:self.check_sim_year(file,start_year,end_year,months),files))
        files_filtered = np.extract(boolean,files)
        dates_filtered = np.extract(boolean,dates)
        # simfile_dict = dict(zip(dates_filtered,files_filtered))
        simfiles = pd.Series(index=dates_filtered,data=files_filtered)
        simfiles = simfiles.sort_index()
        self.simfiles = simfiles
        return 
    def load_single_sim(self,date,reproject_to_mask = True,transform_to_epsg4326 =False,transform_resolution = 0.0002777777):
        file = self.simfiles.loc[date]
        SCA = rioxr.open_rasterio(file).squeeze(drop=True)
        if type(date)==str:
            date = dt.datetime.strptime(date,'%Y-%m-%d')
        SCA['time'] = date
        if reproject_to_mask ==True:
            SCA = self.match_mask(SCA)
        elif transform_to_epsg4326==True:
            SCA = self.match_crs(SCA,transform_resolution)
            # SCA = SCA.rio.reproject(dst_crs="EPSG:4326",resolution = transform_resolution)# 0.001666666)
            # SCA = SCA.rename({'x':'longitude','y':'latitude'})
        return SCA
    def load_all_sims_dask(self,files = None,reproject_to_mask = False,transform_to_epsg4326=True,transform_resolution = 0.0002777777):
        if files ==None:
            files = self.simfiles
        daskarray = xr.open_mfdataset(files,chunks='auto', concat_dim = 'time',combine = 'nested')
        SSM = daskarray.squeeze().band_data
        SSM.name = 'SSM'
        SSM['time'] = files.index
        if reproject_to_mask ==True:
            SSM = self.match_mask(SSM)
        elif transform_to_epsg4326==True:
            SSM = self.match_crs(SSM,transform_resolution)
        # SSM =SSM.rename({'y':'latitude','x':'longitude'})
        self.SSM = SSM
        return self.SSM
    def load_HS_obs(self,station_short_names= None ,
                   SLF_path = "/home/pwiersma/scratch/Data/SLF/envidat_download/study-plot",
                MS_path= "/home/pwiersma/scratch/Data/MeteoSwiss/Snow/hto000d0"):
        if station_short_names == None:
            station_short_names = self.station_short_names
        df_dict = {}
        for name in station_short_names:
            if name[0].isdigit():
                df  = self.load_manual_SLF(name,root = SLF_path)
            else:
                #This includes both MeteoSwiss and SLF stations, basically what I can download from Idaweb
                df =self.load_MS_HS(name,root = MS_path)
            df_dict[name] = df
        self.HS_obs = df_dict 
        return df_dict
    def load_SWE_obs(self,station_short_names= None,
                     root = "/home/pwiersma/scratch/Data/SLF/envidat_download/study-plot"):
        # df_dict = {}
        if station_short_names == None:
            station_short_names = self.station_short_names
        df_list = []
        for name in station_short_names:
            if name in self.SWE_stations:
                data = self.load_manual_SLF(name,root = "/home/pwiersma/scratch/Data/SLF/envidat_download/study-plot")
                if not 'HNW_1D' in data.columns:
                    print(f"Station {name} does not have SWE obs")
                    continue
                SWE = data['HNW_1D'][~np.isnan(data['HNW_1D'])]
                SWE = SWE.rename(name)
                # df_dict[name] = SWE
                df_list.append(SWE)
            else:
                continue
        SWE_df = pd.concat(df_list,axis = 1)
        self.SWE_obs = SWE_df
        return SWE_df
    def match_mask(self,snow_da,mask = None):
        if mask ==None:
            m = self.mask
        else:
            m = mask
        if 'latitude' in snow_da.dims:
            snow_da = snow_da.rename({'longitude':'x','latitude':'y'})
        if type(snow_da)==xr.Dataset:
            for key in snow_da.keys():
                if 'crs' in snow_da.attrs:
                    if 'crs' not in snow_da[key].attrs:
                        snow_da[key].attrs['crs'] = snow_da.attrs['crs']
        new_da = snow_da.rio.reproject_match(m,nodata = np.nan)
        new_da = new_da.rename({'x':'longitude','y':'latitude'})
        return new_da
    def match_crs(self,snow_da,transform_resolution= 0.0002777777):
        da = snow_da.rio.reproject(dst_crs="EPSG:4326",resolution = transform_resolution,nodata = np.nan).rename(
            {'x':'longitude','y':'latitude'})
        # da = xr.where(da>1e30,np.nan,da) #somehow rio adds 3.4e38 on the south and east borders
        da.attrs['crs']= 'epsg:4326'
        return da # 0.001666666)
    def load_MS_HS(self,station_short_name,root = "/home/pwiersma/scratch/Data/MeteoSwiss/Snow/hto000d0"):
        folder = root
        all_files = glob.glob(join(folder,'*hto*d0*data.txt'))
        
    
        # files = {}
        # station_dfs = {}
        files = []
        for file in all_files:
            if station_short_name in file:
                files.append(file)
                
        # sae_files = files['SAE']
        df_list = []
        for f in files:
            if 'autd' in f:
                hto = 'htoautd0'
            else:
                hto = 'hto000d0'
            df = pd.read_csv(f,skiprows = 2,
                               delimiter = ';',
                               index_col = 'time',
                               parse_dates = True,
                               na_values = '-').rename(columns = {'stn':'Station',
                                                   hto:'HS',
                                                   'qhto000d0':'PI',
                                                   'mhto000d0':'MI'})
                                                                  
                                                                  
            df_list.append(df)
        station_df = pd.concat(df_list)                                                      
                                                              
        station_df.attrs['unit'] = 'cm'   
        station_df.attrs['full_name'] = self.long_names[station_short_name]  
        station_df.attrs['coordinates'] = self.station_coordinates[station_short_name]
        station_df.attrs['elevation'] = self.station_elevation[station_short_name] 
        station_df.attrs['jonas2009_region'] = self.jonas2009_region[station_short_name]
        return station_df
    def load_manual_SLF(self,station_short_name,root = "/home/pwiersma/scratch/Data/SLF/envidat_download/study-plot"):
        stations = pd.read_csv(join(root,'stations.csv'),index_col = 'station_code')
        lat, lon = stations.loc[station_short_name,['lat','lon']]
        
        data = pd.read_csv(join(root,f"data/by_station/{station_short_name}.csv"),
                           index_col = 'measure_date',parse_dates = True)
        data['time'] = data.index.date
        data.index = pd.to_datetime(data['time'])
        data.drop(columns = ['time'],inplace = True)
        data = data.rename(columns = {'station_code':'Station'})
        data.attrs['full_name'] = stations.loc[station_short_name,'label']
        data.attrs['coordinates'] = [lon,lat]
        data.attrs['elevation']  = stations.loc[station_short_name,'elevation']
        data.attrs['jonas2009_region'] = int(station_short_name[0])
        data.attrs['unit'] = 'cm'
        return data   
    def jonas2009(self,region = 3, aux_data_dir = '/home/pwiersma/scratch/Data/ewatercycle/aux_data'):
        if not 'HS_obs' in dir(self):
            print('Load HS observations first')
            return
        jonasa = pd.read_csv(join(aux_data_dir,'jonas2009_a.csv'),index_col = 0)
        jonasb = pd.read_csv(join(aux_data_dir,'jonas2009_b.csv'),index_col = 0)
        jonas_offset = pd.read_csv(join(aux_data_dir,'jonas2009_offset.csv'),index_col = 0)
        if not 'HSSWE' in dir(self):
            self.HSSWE = self.HS_obs.copy()        
        for station in self.HSSWE.keys():
            if 'jonas2009_region' in self.HSSWE[station].attrs:
                region = self.HSSWE[station].attrs['jonas2009_region']
            if self.HSSWE[station].attrs['elevation']>2000:
                h_zone = 'h_zone1'
            elif self.HSSWE[station].attrs['elevation']<1400:
                h_zone = 'h_zone3'
            else:
                h_zone = 'h_zone2'
            bulk_density = self.HSSWE[station]['HS']/100 * jonasa.loc[self.HSSWE[station].index.month,h_zone].values + jonasb.loc[self.HSSWE[station].index.month,h_zone].values + jonas_offset.loc[region,'Offset']
            self.HSSWE[station]['SWE_jonas2009'] = bulk_density*self.HSSWE[station]['HS']/100
        return
    def load_modSWE_at_stations(self,resolution = 200,
                             aux_data_dir = '/home/pwiersma/scratch/Data/ewatercycle/aux_data',
                             model_key = None):
        if not 'HS_obs' in dir(self):
            print('Load HS observations first')
            return
        if not 'HSSWE' in dir(self):
            self.HSSWE = self.HS_obs.copy()
        #Depends on resolution
        # comparison_points = pd.read_csv(join(aux_data_dir,f'Thur_HS_gridpoints_{resolution}.csv'),index_col = 0)
        comparison_points = pd.read_csv(join(aux_data_dir,f'JLLRV_HS_gridpoints_1000_noshadow.csv'),index_col = 0)

        for station in self.HSSWE.keys():
            mod_SWE  = self.SWE.sel({'longitude':comparison_points.loc[station,'lon_grid'],
                                    'latitude':comparison_points.loc[station,'lat_grid']},'nearest').to_pandas()
            
            if isinstance(mod_SWE,pd.DataFrame):
                # print('yes is insntance')
                for key in mod_SWE.keys():
                    if not key in ['latitude','longitude','spatial_ref', 'lon','lat']:
                        try :
                            self.HSSWE[station] = self.HSSWE[station].join(mod_SWE[key]).resample('D').mean() 
                        except:
                            self.HSSWE[station] = self.HSSWE[station].join(mod_SWE[key])

                        #This resampling is necessary cause sometimes You get 1000 values for one day which are all the same
            elif isinstance(mod_SWE, pd.Series):
                mod_SWE.name = model_key
                self.HSSWE[station]= self.HSSWE[station].join(mod_SWE, how = 'outer')
        return
    def load_OSHD(self, root = "/home/pwiersma/scratch/Data/SLF/OSHD",resolution = '1000m'):
        OSHD = xr.open_dataset(join(root,f"OSHD_{resolution}_latlon_{self.basin}.nc"))
        OSHD = OSHD.assign_coords(time=OSHD['time'] - pd.to_timedelta(6, unit='h'))
        return OSHD
    def load_FSM(self, root= "/home/pwiersma/scratch/Data/SLF/FSM",resolution = '250m'):
        FSM = xr.open_dataset(join(root,f"FSM_{resolution}_latlon_{self.basin}.nc"))
        FSM = FSM.assign_coords(time=FSM['time'] - pd.to_timedelta(6, unit='h'))
        return FSM
    def load_OSHD_at_stations(self, resolution = 1000,
                              aux_data_dir = '/home/pwiersma/scratch/Data/ewatercycle/aux_data'):
        comparison_points = pd.read_csv(join(aux_data_dir,f'JLLRV_HS_gridpoints_1000_noshadow.csv'),index_col = 0)
        for station in self.HSSWE.keys():
            OSHD = self.load_OSHD()['swee_all'].rename(dict(x = 'longitude',y= 'latitude'))
            OSHD_SWE  = OSHD.sel({'longitude':comparison_points.loc[station,'lon_grid'],
                                    'latitude':comparison_points.loc[station,'lat_grid']},'nearest').to_pandas()
            OSHD_SWE.name = 'OSHD'
            # OSHD_SWE.index =  OSHD_SWE.index- dt.timedelta(hours = 6)
            self.HSSWE[station] = self.HSSWE[station].join(OSHD_SWE, how = 'outer')
        return
    def MS_SWE_catchment_sum(self):
        all_SWE  = pd.DataFrame(None)
        obs_dfs = [self.HSSWE[station]['SWE'] for station in self.HSSWE.keys()]
        all_SWE['MeteoSwiss_SWE'] = d = reduce(lambda x, y: x.add(y, fill_value=0), obs_dfs)
        for key in self.SWE.keys():
            if not key in ['latitude','longitude','spatial_ref']:
                mod_dfs = [self.HSSWE[station][key] for station in self.HSSWE.keys()]
                all_SWE[key] = d = reduce(lambda x, y: x.add(y, fill_value=0), mod_dfs)
        inner_SWE = all_SWE[(all_SWE.MeteoSwiss_SWE>0)&(all_SWE[key]>0)]
        self.MS_SWE_sum = inner_SWE
        return inner_SWE
    # @profile     
    def load_shadow(self,root = "/home/pwiersma/scratch/Data/SnowMaps/Landsat_Shadows/"):
        shadow_files = glob.glob(join(root,"*"))
        # shadow_das = []
        dates = []
        for f in shadow_files:
            # year = os.path.basename(f)[1:5]
            # day = int(os.path.basename(f)[5:8])
            yearday = os.path.basename(f)[1:8]
            dates.append(dt.datetime.strptime(yearday,'%Y%j'))#.strftime('%Y-%m')
            
            # grid = rioxr.open_rasterio(f).squeeze(drop=True).rename({'x':'longitude','y':'latitude'})
            # grid['time'] = date
            # grid = grid.expand_dims(dim = 'time')
            # shadow_das.append(grid)
            
        time_idx = xr.DataArray(data = pd.Index(dates),dims = 'time')
        shadow_ds = xr.open_mfdataset(shadow_files,concat_dim = time_idx,combine = 'nested'
                                      ).squeeze(drop=True
                                                ).rename({'x':'longitude','y':'latitude','band_data':'shadow'}
                                          ).compute()
            
        #TODO use xr.open_mfdataset
        # shadow_ds = xr.concat(shadow_das, dim='time')
        shadow_ds = shadow_ds['shadow'].groupby('time.month').mean() # They're all the the same anyways 
        self.shadow  = shadow_ds
        return self.shadow
    
    def mask_shadow(self,snow_ds):
        #TODO this operation doesn't need to be repeated every time
        # print(self.shadow.dims)
        if (snow_ds.latitude.size != self.shadow.latitude.size) or (snow_ds.longitude.size != self.shadow.longitude.size) :
            print("Shapes do not match, interpolate!")
            shadow = self.shadow.interp_like(snow_ds) 
        else:
            shadow = self.shadow
        for m in range(1,13):
            s_month = shadow.sel({'month':m})            
            snow_ds = xr.where(snow_ds.time.dt.month == m,
                            snow_ds.where(s_month ==1,np.nan),
                            snow_ds)
        snow_ds = snow_ds.drop(['spatial_ref','month'])
        snow_ds.attrs['crs'] = 'epsg:4326'
        return snow_ds
    def compute_fSCA(self, snowcovermap, coarse_grid):
        if 'lat' in coarse_grid.dims:
            coarse_grid = coarse_grid.rename({'lat':'latitude','lon':'longitude'})
        elif 'y' in coarse_grid.dims:
            coarse_grid = coarse_grid.rename({'y':'latitude','x':'longitude'})
        if coarse_grid.rio.crs ==None:
            coarse_grid.rio.write_crs('epsg:4326',inplace = True)
        if coarse_grid.rio.nodata ==None:
            coarse_grid.rio.write_nodata(np.nan,inplace = True,encoded = True)
        
        if snowcovermap.rio.crs ==None:
            snowcovermap.rio.write_crs('epsg:4326',inplace = True)
        if snowcovermap.rio.nodata ==None:
            snowcovermap.rio.write_nodata(np.nan,inplace = True,encoded = True)
        # coarse_grid_withzeros = coarse_grid.fillna(0)

        fsca = snowcovermap.rio.reproject_match(coarse_grid,
                                                resampling = Resampling.average,
                                                nodata = np.nan).rename({'y':'latitude','x':'longitude'})
        #kwargs = dict(fill_value = 'extrapolate') didn't really work, the southern-most pixel is still missing 
        fsca = xr.where(np.isnan(coarse_grid),np.nan,fsca)
        return fsca
    def load_SLF_SWE(self,station_short_names= None,
                     # station_names_short_SLF,
                     root = "/home/pwiersma/scratch/Data/SLF/from_Tobias"):
        if not 'HS_obs' in dir(self):
            print('Load HS observations first')
            return
        if not 'HSSWE' in dir(self):
            self.HSSWE = self.HS_obs.copy()
        
        # if self.basin =='Jonschwil':
        SLF_SWE_1 = pd.read_csv(os.path.join(root,"PWiersma.OSHD.SWE.txt"),
                                  skiprows = 22,delimiter = '\t',skip_blank_lines=True,
                                parse_dates={'date': ['Year','Month','Day']},index_col = 'date')
        SLF_SWE_2 = pd.read_csv(os.path.join(root,"PWiersma_oct2023.OSHD.SWE.txt"),
                                  skiprows = 22,delimiter = '\t',skip_blank_lines=True,
                                parse_dates={'date': ['Year','Month','Day']},index_col = 'date')
        SLF_SWE_3 = pd.read_csv(os.path.join(root,"SLF_SWE_Riale_di_Calneggia.csv"),
                            parse_dates = True, index_col = 0)
        SLF_SWE_3 = SLF_SWE_3.rename_axis('time', axis='index')
        SLF_SWE_3.index = SLF_SWE_3.index.strftime('%Y-%m-%d')
        SLF_SWE_3.index = pd.to_datetime(SLF_SWE_3.index)

        SLF_SWE_1.replace(-999,np.nan,inplace =True)
        SLF_SWE_2.replace(-999,np.nan,inplace =True)
        

        
        # for i in range(len(station_names_short_MS)):
        if station_short_names ==None:
            station_short_names = self.station_short_names
            
        for name in station_short_names:
            if name in ['3UI0','5VZ0','5PU0','5SA0','5KR0','5DO0']:
                SLF_SWE = pd.read_csv(join(root,'generated_myself',f"{name}_fromTobiasScript.csv"),
                                      index_col = 'Row',parse_dates = True)
                self.HSSWE[name]['SWE'] = SLF_SWE['SWE']
                
            else:
                SLF_short_name = self.slf_short_names[name]
                if name in ['SLF3SW','WHA','MGB','SPZ']:
                    SLF_SWE = SLF_SWE_1
                elif name in ['6BG0','6RO0','SLFBO2']:
                    SLF_SWE = SLF_SWE_3
                else:
                    SLF_SWE = SLF_SWE_2
                # print(name)
                if name =='LAN':
                    # LAN = SLF_SWE['MCH.LAN2']
                    # concat = pd.concat(SLF_SWE)
                    SLF_SWE['MCH.LAN2'].fillna(SLF_SWE['SLF.5LQ'], inplace = True)
                elif name == 'BUF':
                    SLF_SWE['MCH.BUF2'].fillna(SLF_SWE['SLF.7BU'], inplace = True)
                print(name, SLF_short_name)
                self.HSSWE[name]['SWE'] = SLF_SWE[SLF_short_name]
            


            
            # self.HSSWE[name][['SWE','HS']][slice('2012','2015')].plot(figsize = (12,5),title =name)
            # plt.ylabel('[cm]')
        # .rename(columns = 
                                # {'SLF.3SW':'SLF3SW',
                                #  'MCH.WHA2':'WHA',
                                #  'MCH.MGB2':'MGB',
                                #  'MCH.SPZ2':'SPZ'})
        
        # coarse_grid = coarse_grid = staticmaps.wflow_dem
        # coarse_grid = coarse_grid.rename({'lon':'longitude','lat':'latitude'})
        
        # sc= masked_landsat.snowcover
        # # sc = sc.rio.
        # sc= xr.where(np.isnan(sc.data),np.nan,sc)
        # sc.rio.write_crs('epsg:4326',inplace=True)
        # sc.rio.write_nodata(np.nan,inplace = True,encoded=True)
        
        # fsca_list = []
        # for t in sc.time[:20]:
        #     fsca = sc.sel({'time':t}).rio.reproject_match(coarse_grid,
        #                                                            resampling = Resampling.average,
        #                                                            nodata = np.nan)
        #     fsca_masked = xr.where(np.isnan(coarse_grid.data),np.nan, fsca)
        #     fsca_list.append(fsca_masked)
        # fsca_concat =xr.concat(fsca_list,dim = 'time')
        
        
        # #Count the amount of nans and then remove them (nans are 1e9 in this case)
        # resampled_sum = sc0.rio.reproject_match(coarse_grid, resampling = Resampling.average,nodata = np.nan)
        # nancount = resampled_sum.round(-4)/1e12 #there's a maximum of 1111 values per coarse res grid cell
        # residual = resampled_sum%1e4
        
        
        # sc_clip = sc.sel({'longitude':slice(9.05,9.20),'latitude':slice(47.3,47.25)})[0]
        # clip_reproject = sc_clip.rio.reproject(dst_crs = sc_clip.rio.crs, resolution = 0.007,resampling = Resampling.average,nodata = np.nan)
        
    def calculate_benchmark(self, station_short_name , SWE_column = 'SWE', include_unreduced_benchmark = False,
                            mask_nonzero_summer=True):
        print(station_short_name)
        SWE = self.HSSWE[station_short_name][[SWE_column, 'hyear']]
        SWE['DOY'] = SWE.index.day_of_year
        
        SWE_DOY =  SWE[SWE_column].groupby(SWE.DOY).mean()
        SWE_merged = SWE.merge(SWE_DOY, on= 'DOY', how = 'left').rename(columns = {f"{SWE_column}_x":SWE_column,
                                                                                   f"{SWE_column}_y":f"{SWE_column}_DOY"})
        SWE_merged.index = SWE.index
        
        SWE.pop('DOY')
        if SWE.columns.size != 2:
            print(f"Something is going wrong in station {station_short_name}")
        SWE_hyearmeans = SWE.groupby('hyear').mean()
    
        SWE_merged[f"{SWE_column}_DOY_reduced"] = SWE_merged[f"{SWE_column}_DOY"]
        for hyear in SWE_hyearmeans.index:
            reduction = SWE_DOY.mean() - SWE_hyearmeans[SWE_hyearmeans.index ==hyear].values[0][0]
            SWE_merged.loc[SWE_merged['hyear']==hyear,f"{SWE_column}_DOY_reduced"] = \
                SWE_merged.loc[SWE_merged['hyear'] == hyear, f"{SWE_column}_DOY"]- reduction 
        # if mask_nonzero_summer==True:
        #     SWE_merged.loc[(SWE_merged[SWE_column]==0) & (SWE_merged[f"{SWE_column}_DOY"].diff()<10),
        #                    [f"{SWE_column}_DOY",f"{SWE_column}_DOY_reduced"]] =0
        if include_unreduced_benchmark ==True:
            self.HSSWE[station_short_name][f"{SWE_column}_DOYmean"] = SWE_merged[f"{SWE_column}_DOY"]
        self.HSSWE[station_short_name][f"{SWE_column}_DOYcorrected"] = SWE_merged[f"{SWE_column}_DOY_reduced"]
    def calculated_benchmarks(self, station_short_names=None, SWE_column = 'SWE', include_unreduced_benchmark=False):
        if not 'HSSWE' in dir(self):
            print('HSSWE not present, load it first')
            return
        if station_short_names ==None:
            station_short_names = self.station_short_names
        for name in station_short_names:
            if not 'hyear' in self.HSSWE[name].columns:
                self.add_hydro_year_column(self.HSSWE[name])
            self.calculate_benchmark(name, SWE_column =SWE_column , include_unreduced_benchmark=include_unreduced_benchmark)
    def add_hydro_year_column(self,df):
        """
        Add a hydrological year column to a DataFrame with a daily datetime index.
    
        Parameters:
        ----------
        df : pandas DataFrame
            The DataFrame with a daily datetime index to which the hydrological year column will be added.
    
        Returns:
        ----------
        pandas DataFrame
            The DataFrame with the hydrological year column added.
        """
        # Extract year and month from the index
        df['Year'] = df.index.year
        df['Month'] = df.index.month
    
        # Define the start month and end month of the hydrological year
        start_month = 10  # October
        end_month = 9     # September
    
        # Create a mask to identify the days within the hydrological year
        # mask = (df['Month'] >= start_month) | (df['Month'] <= end_month)
    
        # Increment the year for days before October
        df['hyear'] = df['Year'] 
        df.loc[df['Month']>=10,'hyear'] += 1
        
        # df.loc[~mask, 'hyear'] = df['Year']
    
        # Remove the temporary 'Year' and 'Month' columns
        df.drop(['Year', 'Month'], axis=1, inplace=True)
    
        return df
        
# shadow_folder = "/home/tesla-k20c/data/pau/SnowMaps/Landsat_Shadows/"
# shadow_files = glob.glob(join(shadow_folder,"*"))
# shadow_das = []
# for f in shadow_files:
#     # year = os.path.basename(f)[1:5]
#     # day = int(os.path.basename(f)[5:8])
#     yearday = os.path.basename(f)[1:8]
#     date  = dt.datetime.strptime(yearday,'%Y%j')#.strftime('%Y-%m')
#     grid = rioxr.open_rasterio(f).squeeze(drop=True).rename({'x':'longitude','y':'latitude'})
#     grid['time'] = date
#     grid = grid.expand_dims(dim = 'time')
#     shadow_das.append(grid)
# shadow_ds = xr.concat(shadow_das, dim='time')
# # date = pd.to_datetime(day,unit='D',origin = year).strftime('%Y-%m')
    
    
    
    
# C=SnowClass()
# C.load_mask("/home/pwiersma/scratch/Data/ewatercycle/aux_data")
# # C.load_dem("/home/pwiersma/scratch/Data/ewatercycle/aux_data/wflow_dem.map")
# C.load_SWE("/home/pwiersma/scratch/Data/ewatercycle/output_dir/Snowmaps/SWE_wflow_1993_2019_JS_200m.nc")
# C.SWE_to_snowcover(threshold=5)
# C.load_simulation_files("/home/tesla-k20c/data/pau/SnowMaps/SimulatedSnowCover",2002,2004)
# single_sim = C.load_single_sim(C.simfiles.index[0],reproject_to_mask=False,transform_to_epsg4326=True)
# # single_sim_coarse = C.match_mask(single_sim)
# C.load_landsat("/home/tesla-k20c/data/pau/SnowMaps/RealLandsatSnowCover",reproject_to_mask = False,transform_to_epsg4326=True)
# HSSWE = C.load_MeteoSwiss(station_short_names = ['RIC','MGB','RHO','SPZ','WHA','SAE','SLF3SW','SWA'])   
# C.load_shadow()

# SLF_SWE = pd.read_csv("/home/tesla-k20c/data/pau/SLF/from_Tobias/PWiersma.OSHD.SWE.txt",
#                       skiprows = 22,delimiter = '\t',skip_blank_lines=True,
#                     parse_dates={'date': ['Year','Month','Day']},index_col = 'date').rename(columns = 
#                         {'SLF.3SW':'SLF3SW',
#                          'MCH.WHA2':'WHA',
#                          'MCH.MGB2':'MGB',
#                          'MCH.SPZ2':'SPZ'})
# station_short_names = ['SLF3SW','MGB','WHA','SPZ']
# for name in station_short_names:
#     C.HSSWE[name]['SWE'] = SLF_SWE[name]/10
#     C.HSSWE[name][['SWE','HS']][slice('2012','2015')].plot(figsize = (12,5),title =name)
#     plt.ylabel('[cm]')
# masked_ls = C.mask_shadow(C.landsat)
# masked_single_sim = C.mask_shadow(single_sim)



# sh = C.shadow.interp_like(single_sim)
# single_sim_month = pd.to_datetime(single_sim.time.data).month
# sh = sh.sel({'month':single_sim_month})
# masked_sim = xr.where(sh==0,np.nan,single_sim)
# sh = C.shadow.rio.reproject_match(single_sim)
# sh = xr.where(sh>1e30,np.nan,sh)
# sh = C.shadow.rename({'x':'longitude','y':'latitude'})
    

    
    
    # def resample_like(self, source, dest, method):
        
    
        # return
        # self.dates = dates
        # return
        # simsnow_list_all     = []
        # for file in sim_paths:
        #     tif = rioxr.open_rasterio(file)
        #     simdate = get_simulation_date(file)
        #     if (simdate.month>10)|(simdate.month<5):
        #         tif = tif.assign_attrs({'time':simdate}).squeeze(dim='band',drop=True)
        #         tif['time'] = get_simulation_date(file)
        #         simsnow_list_all.append(tif)
        # simsnow_all = xr.concat(simsnow_list_all,dim='time').sortby('time').to_dataset(name='snowcover')
        # simsnow_all['ls_bool']=(('time'),np.zeros(len(simsnow_all.time),dtype='bool'))



    
    # def check_sim_year(file):
    #     simdate = self.get_simulation_date(file)
    #     if (simdate.year==year)&(simdate.month==3):
    #         return True
    #     else:
    #         return False


#%%


    


# # wfsnow_path = "N:/PauWiersma/GIS/test_snow_200m_JS.nc"
# wfsnow_path = "/home/pwiersma/scratch/Data/ewatercycle/output_dir/Snowmaps/SWE_wflow_1993_2019_JS_200m.nc"
# wfsnow_Jonschwil     = xr.open_dataset(wfsnow_path).Snow
# # wfsnow = wfsnow_Jonschwil.where(~xr.ufuncs.isnan(mask),-999)

# mask = rioxr.open_rasterio("/home/pwiersma/scratch/Data/ewatercycle/aux_data/Jonschwil_Mask.tif")
# mask = mask.rename({'x':'longitude','y':'latitude'}).squeeze(dim='band',drop=True)
# mask = mask.sortby('latitude',ascending=True)
# mask = mask.sortby('longitude',ascending=True)
# mask = mask.interp_like(wfsnow_Jonschwil,method='nearest')

# wfsnow = wfsnow_Jonschwil.where(~xr.ufuncs.isnan(mask),drop=True)

# #%% Load Landsat observations
# year=2006

# def get_landsat_date(Landsat_file):
#     basename= os.path.basename(Landsat_file).split('.')[0]
#     datestring = basename.split('_')[3]
#     datetime    = dt.datetime.strptime(datestring,'%Y%m%d')
#     return datetime



# def check_year(file):
#     if get_landsat_date(file).year==year:
#         return True
#     else:
#         return False

# lssnow_files = glob.glob("/home/tesla-k20c/data/pau/SnowMaps/RealLandsatSnowCover/*")
# lssnow_dates = list(map(get_landsat_date,lssnow_files))
# lssnow_filtered = list(filter(check_year,lssnow_files))

# lssnow_list      = []
# for file in lssnow_filtered:
#     tif = rioxr.open_rasterio(file)
#     tif = tif.assign_attrs({'time':get_landsat_date(file)})
#     tif['time'] = get_landsat_date(file)
#     lssnow_list.append(tif)
# lssnow = xr.concat(lssnow_list,dim='time').squeeze(
#     dim='band',drop=True).sortby('time')

# lssnow_all_list = []
# for file in lssnow_files:
#     tif = rioxr.open_rasterio(file)
#     lsdate = get_landsat_date(file)
#     if (lsdate.month>10)|(lsdate.month<5):
#         tif = tif.assign_attrs({'time':lsdate})
#         tif['time'] = get_landsat_date(file)
#         lssnow_all_list.append(tif)
# lssnow_all = xr.concat(lssnow_all_list,dim='time').squeeze(
#     dim='band',drop=True).sortby('time').to_dataset(name='snowcover')
# lssnow_all['ls_bool']=(('time'),np.ones(len(lssnow_all.time),dtype='bool'))
    
# # Simulat#%% Compare with Observations

# wfsnow_filtered = wfsnow.sel({'time':lssnow.time})
# wfsnowcover = xr.where(wfsnow_filtered>1,1,0).where(~xr.ufuncs.isnan(wfsnow[0]),np.nan)


# # for image in lssnow:
# #     image.plot()
# #     plt.show()
# # for image in wfsnowcover:
# #     image.plot()
# # #     plt.show()



# for i in range(len(lssnow)):
#     ls = lssnow[i]
#     wf = wfsnowcover.sel({'time':ls.time})
#     f1,(ax1,ax2) = plt.subplots(1,2,figsize=(7,4))
#     plt.subplots_adjust(wspace=0.05)
#     ls.plot(ax=ax1,cmap='binary_r',add_colorbar=False)
#     wf.plot(ax=ax2,cmap='binary_r',add_colorbar=False)
#     ax1.get_xaxis().set_visible(False)
#     ax2.get_xaxis().set_visible(False)
#     ax1.get_yaxis().set_visible(False)
#     ax2.get_yaxis().set_visible(False)
#     ax1.set_title('Landsat snow cover (30m)')
#     ax2.set_title('wflow snow cover (200m)')
#     f1.suptitle(np.datetime_as_string(ls.time,unit='D'))
#     ax1.set_facecolor('wheat')
#     ax2.set_facecolor('wheat')
    

# #%% Compare with simulations

# sim_paths = glob.glob("/home/tesla-k20c/data/pau/SnowMaps/SimulatedSnowCover/*")

# def get_simulation_date(Landsat_file):
#     basename= os.path.basename(Landsat_file).split('.')[0]
#     datetime    = dt.datetime.strptime(basename,'%Y%m%d')
#     return datetime

# def check_sim_year(file):
#     if (get_simulation_date(file).year==year)&(get_simulation_date(file).month==3):
#         return True
#     else:
#         return False
# sim_dates = list(map(get_simulation_date,sim_paths))
# simsnow_filtered = list(filter(check_sim_year,sim_paths))

# simsnow_list      = []
# for file in simsnow_filtered:
#     tif = rioxr.open_rasterio(file)
#     tif = tif.assign_attrs({'time':get_simulation_date(file)})
#     tif['time'] = get_simulation_date(file)
#     simsnow_list.append(tif)
# simsnow = xr.concat(simsnow_list,dim='time').squeeze(
#     dim='band',drop=True).sortby('time')

# simsnow_list_all     = []
# for file in sim_paths:
#     tif = rioxr.open_rasterio(file)
#     simdate = get_simulation_date(file)
#     if (simdate.month>10)|(simdate.month<5):
#         tif = tif.assign_attrs({'time':simdate}).squeeze(dim='band',drop=True)
#         tif['time'] = get_simulation_date(file)
#         simsnow_list_all.append(tif)
# simsnow_all = xr.concat(simsnow_list_all,dim='time').sortby('time').to_dataset(name='snowcover')
# simsnow_all['ls_bool']=(('time'),np.zeros(len(simsnow_all.time),dtype='bool'))


# wfsnow_march = wfsnow.sel({'time':simsnow.time})
# wfsnowcover_march = xr.where(wfsnow_march>1,1,0).where(~xr.ufuncs.isnan(wfsnow[0]),np.nan)

# for i in range(len(simsnow)):
#     ls = simsnow[i]
#     wf = wfsnowcover_march.sel({'time':ls.time})
#     f1,(ax1,ax2) = plt.subplots(1,2,figsize=(7,4))
#     plt.subplots_adjust(wspace=0.05)
#     im1 = ls.plot(ax=ax1,cmap='binary_r',add_colorbar=False,vmin=0,vmax=1)
#     im2 = wf.plot(ax=ax2,cmap='binary_r',add_colorbar=False,vmin=0,vmax=1)
#     ax1.get_xaxis().set_visible(False)
#     ax2.get_xaxis().set_visible(False)
#     ax1.get_yaxis().set_visible(False)
#     ax2.get_yaxis().set_visible(False)
#     ax1.set_title('Synthetic snow cover (30m)')
#     ax2.set_title('wflow snow cover (200m)')
#     f1.suptitle(np.datetime_as_string(ls.time,unit='D'))
#     ax1.set_facecolor('wheat')
#     ax2.set_facecolor('wheat')


# #%%
# #winter_snowmaps = xr.concat([simsnow_all.sel({'time':slice('2010','2019')}),
#                              # lssnow_all.sel({'time':slice('2010','2019')})],dim='time').sortby('time')#.to_dataset(name='snowcover')
# # winter_snowmaps = winter_snowmaps.sel({'time':slice('2000','2019')})


# #winter_snowmaps.snowcover.attrs['time']='nu ff niet'
# #winter_snowmaps.to_netcdf('D:/PhDs/PauWiersma/SC_LS+sim_2000_2019.nc')

# winter_snowmaps = xr.open_dataset('D:/PhDs/PauWiersma/SC_LS+sim_2000_2019.nc')



# winter_wfSWE = wfsnow.sel({'time':winter_snowmaps.time})
# winter_wfcover = xr.where(winter_wfSWE>1,1,0).where(~xr.ufuncs.isnan(wfsnow[0]),np.nan)


# #%% Animation
# # imgs = []

# # for i in range(len(winter_wfcover)):
# #     ls = winter_snowmaps.snowcover[i]
# #     wf = winter_wfcover.sel({'time':ls.time})
# #     f1,(ax1,ax2) = plt.subplots(1,2,figsize=(7,4))
# #     plt.subplots_adjust(wspace=0.05)
# #     im1 = ls.plot(ax=ax1,cmap='binary_r',add_colorbar=False,vmin=0,vmax=1)
# #     im2 = wf.plot(ax=ax2,cmap='binary_r',add_colorbar=False,vmin=0,vmax=1)
# #     ax1.get_xaxis().set_visible(False)
# #     ax2.get_xaxis().set_visible(False)
# #     ax1.get_yaxis().set_visible(False)
# #     ax2.get_yaxis().set_visible(False)
# #     ax2.set_title('wflow snow cover (1km)')
# #     f1.suptitle(np.datetime_as_string(ls.time,unit='D'))
# #     if winter_snowmaps.ls_bool[i]==True:
# #         ax1.set_title('Landsat snow cover (30m)')
# #         ax1.patch.set_facecolor('blueviolet') #not plum
# #         ax1.patch.set_alpha(0.18)
# #         ax2.patch.set_facecolor('blueviolet')
# #         ax2.patch.set_alpha(0.18)
# #     else:
# #         ax1.set_title('Synthetic snow cover (30m)')
# #         # ax1.set_facecolor('lightseagreen',alpha=0.3)
# #         ax1.patch.set_facecolor('lightseagreen')
# #         ax1.patch.set_alpha(0.18)
# #         ax2.patch.set_facecolor('lightseagreen')
# #         ax2.patch.set_alpha(0.18)
# #     datestring = pd.to_datetime(ls.time.to_pandas()).strftime('%Y_%m_%d')
# #     plt.savefig('D:/PhDs/PauWiersma/Figures/formovie_1km/comparison_'+datestring+'_1km.png')
# #     plt.close(f1)


# #%%
# # import imageio
# # figpaths = glob.glob("D:/PhDs/PauWiersma/Figures/forAGU/*2016*")
# # ims = [imageio.imread(f) for f in figpaths]#[:363]]
# # kwargs = {'fps':1}
# # imageio.mimwrite("D:/PhDs/PauWiersma/Figures/testmovie.mp4", ims,'mp4',**kwargs)



# #%% For AGU presentation
# # wf2016_1km = winter_wfcover.loc['2016-03']
# wf2016_200m = winter_wfcover.loc['2016-03']
# ls2016 = winter_snowmaps.sel({'time':'2016-03'})

# for i in range(len(wf2016_1km)):
#     ls = ls2016.snowcover[i]
#     wf1 = wf2016_1km.sel({'time':ls.time})
#     wf200= wf2016_200m.sel({'time':ls.time})
#     f1,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,4),dpi=400)
#     plt.subplots_adjust(wspace=0.05)
#     im1 = ls.plot(ax=ax1,cmap='binary_r',add_colorbar=False,vmin=0,vmax=1)
#     im2 = wf200.plot(ax=ax2,cmap='binary_r',add_colorbar=False,vmin=0,vmax=1)
#     im3 = wf1.plot(ax=ax3,cmap='binary_r',add_colorbar=False,vmin=0,vmax=1)
#     ax1.get_xaxis().set_visible(False)
#     ax2.get_xaxis().set_visible(False)
#     ax3.get_xaxis().set_visible(False)
#     ax1.get_yaxis().set_visible(False)
#     ax2.get_yaxis().set_visible(False)
#     ax2.set_title('wflow snow cover (200m)')
#     ax3.get_yaxis().set_visible(False)
#     ax3.set_title('wflow snow cover (1km)')
#     f1.suptitle(np.datetime_as_string(ls.time,unit='D'))
#     if ls2016.ls_bool[i]==True:
#         ax1.set_title('Landsat snow cover (30m)')
#         ax1.patch.set_facecolor('blueviolet') #not plum
#         ax1.patch.set_alpha(0.18)
#         ax2.patch.set_facecolor('blueviolet')
#         ax2.patch.set_alpha(0.18)
#         ax3.patch.set_facecolor('blueviolet')
#         ax3.patch.set_alpha(0.18)
#     else:
#         ax1.set_title('Synthetic snow cover (30m)')
#         # ax1.set_facecolor('lightseagreen',alpha=0.3)
#         ax1.patch.set_facecolor('lightseagreen')
#         ax1.patch.set_alpha(0.18)
#         ax2.patch.set_facecolor('lightseagreen')
#         ax2.patch.set_alpha(0.18)
#         ax3.patch.set_facecolor('lightseagreen')
#         ax3.patch.set_alpha(0.18)
#     datestring = pd.to_datetime(ls.time.to_pandas()).strftime('%Y_%m_%d')
#     plt.savefig('D:/PhDs/PauWiersma/Figures/forAGU/comparison_'+datestring+'_200m_1km_long.png',dpi=400)
#     plt.close(f1)





# ###########3JUNK
# # names = list(map(os.path.basename,lssnow_files))
# # splits = list(map(lambda name:name.split('.')[0],names))
# # years =  list(map(lambda name:name.split('_')[3][:4],splits))
# # idx     = list(np.array(years)==str(year))
# # lssnow_files = list(np.array(lssnow_files)[idx])

#%%
# short_names = ['MGB','WHA','SPZ','SLF3SW','LAN','KUB','SLFKL2','SLFPAR','SLFKL3','SLFPAR','SLFKL3',
#                 'DMA','SLFDA3','SLFWFJ','SLFSLF','SLFFLU','SLFDA2',
#                 'MUS','SLFOF2','SON','SLFFRA'] #SLFDTR and DAV4 missing
# C = SnowClass()
# for name in short_names:
#     print(name)
#     data = C.load_HSSWE(name)
#     hs = data['HS']
#     plt.figure()
#     hs.plot()
#     plt.title(name)
#     plt.show()
    
#%% load SWE obs
# SWE_stations_short = ['3UI0','3SW0','5PU0','5SA0','5KK0','5WJ0','5DF0','5DO0','5MA0','7ST0']
C = SnowClass('Riale_di_Calneggia')
# C.load_SWE_obs(SWE_stations_short)