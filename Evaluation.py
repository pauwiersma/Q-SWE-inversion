

import os
import pandas as pd
from os.path import join
from SnowClass import *
from wflow_spot_setup import *
# from hydrographs_fromdic import *
import seaborn as sns
import matplotlib.pyplot as plt
from flexitext import flexitext
from matplotlib import patches
# from pypalettes import load_cmap
from rasterio.mask import mask
from matplotlib.colors import LightSource
from rasterio.plot import show
import math
import rasterio 
import geopandas as gpd
from typing import Tuple,Union
from matplotlib.dates import DateFormatter 

# os.chdir("/home/pwiersma/scratch/Scripts/Github/ewc/utils")
# from hydrographs_fromdic import *


config_dir = "/home/pwiersma/scratch/Data/ewatercycle/experiments/config_files"
# # config_dir="/work/FAC/FGSE/IDYST/gmariet1/gaia/pwiersma/ewatercycle/experiments/config_files"

# EXP_ID = 'Syn_1811'
# cfg = join(config_dir, f"{EXP_ID}_config.json")
# with open(cfg, 'r') as f:
#     config = json.load(f)
# config['BASIN'] = "Dischma"


# E = Evaluation(config)
# E.load_Q()
# E.load_OFs()

# E.load_SWE()
# E.load_SWEbands()
# E.load_pars()
# E.plot_pars()
# E.load_scalaroutput()
# E.compute_yearly_scalars()
# E.compute_daily_scalars()

# ilats = [9,3,3,3]
# ilons = [2,2,6,10]
# # ilat = 10
# # ilon = 5
# timerange = slice('2009-10-01','2010-09-30')
# for ilat,ilon in zip(ilats,ilons):
#     E.plot_pixel_scalars(ilon = ilon, ilat = ilat,timerange =timerange)

# # E.plot_OF()
# if not config['SWE_CALIB']:
#     E.hydrographs_soilcalib()
#     E.hydrographs_yearlycalib()
# E.plot_swebands() 
# # E.plot_waterbalance()
# E.composite_plot()
# if config['SETTING'] == 'Real':
#     # E.plot_each_station()
#     E.stations_OF()
#     E.load_SC()
#     E.plot_OA()


# Synobs_f = "/home/pwiersma/scratch/Data/ewatercycle/experiments/data/output_Synthetic_obs_Dischma_Syn_1610/output_Synthetic_obs_dischma_Syn_1610.csv"
# Synobs_scalars = pd.read_csv(Synobs_f, index_col=0, parse_dates=True)
# Syearly = Synobs_scalars.resample('Y').sum()
# imbalance = Syearly['Pmean']/100 - Syearly['et'] - Syearly['Q']

# satimbalance = Synobs_scalars['satstore'].resample('Y').first()[1:].diff()
# unsatimbalance = Synobs_scalars['unsatstore'].resample('Y').first()[1:].diff()
# plt.figure()
# plt.plot(satimbalance,label = 'satimbalance')
# plt.plot(unsatimbalance,label = 'unsatimbalance')
# plt.plot(imbalance,label = 'imbalance fluxes')
# plt.axhline(0,color = 'black',linestyle = 'dashed')
# plt.grid()
# plt.legend()





def normalize_swe(MODSWE):
    # Calculate the mean SWE for each day
    MODSWE_mean = MODSWE.mean(axis=1).to_frame('SWEmean')
    MODSWE_mean['DOY'] = MODSWE.index.day_of_year

    # Group by day of year and calculate the mean
    MODSWE_DOY = MODSWE_mean.groupby('DOY').mean()
    MODSWE_DOY.index.name = 'DOY'

    # Merge the daily mean with the original data
    SWE_norm = MODSWE_mean.merge(MODSWE_DOY, on='DOY', how='left')['SWEmean_y']
    SWE_norm.index = MODSWE.index

    # Normalize the SWE values
    SWE_normed = MODSWE.div(SWE_norm, axis=0)
    
    return SWE_normed

def m3s_to_mm(daily_m3s, area):
    # Convert m3/s to mm/day
    daily_mm = daily_m3s * 1000 * 86400 / (area*1e6)
    return daily_mm

def mm_to_m3s(daily_mm, area):
    # Convert mm/day to m3/s
    daily_m3s = daily_mm *1e-3 * (1/86400) * (area*1e6)
    return daily_m3s

def snowvars2swe(snow,snowwater):
    totalsnow = xr.Dataset()
    for var in snow.variables:
        if var in ['lat','lon','time']:
            continue
        short_var = var[5:]
        totalsnow[short_var] = snow[var] + snowwater[f'snowwater_{short_var}']
    return totalsnow

class Evaluation(object):
    """
    Should contain
        - load Q
        - Hydrograph plotting
            - 
        - Loading of MODIS
        - Loading of station data
        - Loading of C snow class 
    """
    def __init__(self, 
                 config,
                 palette = 'colorblind'):
        self.cfg = config
        for key, value in config.items():
            setattr(self, key, value)
        for key,value in config['OBS_ERROR'].items():
            setattr(self,key,value)
        self.palette = palette
        self.colors = {'prior':'tab:orange',
                       'posterior':'tab:blue',
                            'obs':'black',
                            'OSHD':'tab:red',
                            'Naive':'saddlebrown',
                            'Synthetic':'black'}
        self.alphas = dict(prior = 0.2,
                           posterior = 0.5,
                           obs = 1,
                           OSHD = 0.7,
                           Naive = 0.7,
                           Synthetic = 1)
        self.names = dict(prior = 'QSWE-prior',
                          posterior = 'QSWE-posterior',
                          obs = 'Observations',
                          OSHD = 'OB',
                          Naive = 'NB', 
                          Synthetic = 'Synthetic')
        self.colors2 = {name: self.colors[key] for key, name in self.names.items()}
        self.alphas2 = {name: self.alphas[key] for key, name in self.names.items()}

        self.FIGDIR = join(self.OUTDIR,'Figures')
        self.actualres = int(self.RESOLUTION[:-1]) *8.333e-06 
        self.LATITUDE = SwissStation(self.BASIN).lat
        self.grid_area = self.compute_grid_area()
        if self.CONTAINER in [None,'' ]:
            dem = xr.open_dataset(join(self.ROOTDIR,'wflow_staticmaps',f'staticmaps_{self.RESOLUTION}_{self.BASIN}_feb2024.nc'))['wflow_dem']
        else:
            dem = xr.open_dataset(join(self.CONTAINER,f"staticmaps_{self.RESOLUTION}_{self.BASIN}_feb2024.nc"))['wflow_dem']
        self.dem = dem.sortby('lat',ascending=True).sortby('lon',ascending=True)
        self.dem_area = np.sum(dem.notnull()).item() * self.grid_area

        if not os.path.exists(self.FIGDIR):
            os.makedirs(self.FIGDIR)
        
    def compute_grid_area(self):
        lat_deg = float(self.actualres)
        lon_deg = float(self.actualres)
        lat_km = 111.32 * lat_deg
        lon_km = 111.32 * lon_deg * math.cos(math.radians(self.LATITUDE))
        grid_area = lat_km * lon_km
        return grid_area
        
    def load_Q(self):
        self.load_Q_soilcalib()
        self.load_Q_yearlycalib()
        if self.SETTING == 'Real':
            self.load_Q_Naive()
            self.load_Q_OSHD()
        elif self.SETTING == 'Synthetic':
            self.load_Q_Synthetic()
    
    # def load_spotpy_Q(self, calibration_purpose):
    #     import spotpy.analyser as spa
    #     if calibration_purpose =='Yearlycalib':
    #         spQ = pd.DataFrame()
    #         for year in range(self.START_YEAR,self.END_YEAR+1):
    #             for ksoil in range(self.SOIL_K):
                    


    #                 spotpyfile = join(self.OUTDIR,'Yearlycalib',str(year),str(ksoil),f"{self.BASIN}_Q.csv")
    #                 spQ_k = spa.csv_to_dataframes(spotpyfile)
    #                 spQ_k['year'] = year
    #                 spQ_k['ksoil'] = ksoil
    #                 spQ = pd.concat([spQ,spQ_k])

    def load_Q_soilcalib(self):
        df = pd.read_csv(join(self.SOILDIR,
                                  f"{self.BASIN}_Q.csv"),
                                  index_col = 0, parse_dates = True)
        self.Q_soil = {}
        self.Q_soil['obs'] = df['obs']
        for dist in ['prior','posterior']:
            cols = [col for col in df.columns if dist in col]
            self.Q_soil[dist] = df[cols]
        # print('Q_soil prior',self.Q_soil['prior'].mean())

    def load_Q_yearlycalib(self):
        self.Q_yearly = {}
        for dist in ['prior','posterior']:
            Q_yearly_list = []
            Q_obs_list = []
            for year in range(self.START_YEAR,self.END_YEAR+1):
                Q_klist = []
                for ksoil in range(self.SOIL_K):
                    Q_kk = pd.read_csv(join(self.YEARLYDIR,
                                                                f"{year}",
                                                                f"{ksoil}",
                                                                f"{self.BASIN}_Q.csv"),
                                                            index_col = 0, parse_dates = True)
                    Q_klist.append(Q_kk)
                Q_k = pd.concat(Q_klist, axis = 1)
                Q_k = Q_k.loc[:,~Q_k.columns.duplicated()]



                Q_obs_list.append(Q_k['obs'])

                cols = ['obs']+[col for col in Q_k.columns if dist in col]
                Q_k = Q_k[cols]
                Q_k.columns = ['obs']+[col.split('_')[-2] + '_' + col.split('_')[-1] for col in Q_k.columns[1:]]
                # Q_k.columns = ['obs']+[col.split('_')[-2] + '_' + col.split('_')[-1] for col in Q_k.columns[1:]]

                Q_yearly_list.append(Q_k)
            Q_yearly = pd.concat(Q_yearly_list)
            Q_obs = pd.concat(Q_obs_list)
            self.Q_yearly[dist] = Q_yearly
            self.Q_yearly['obs'] = Q_obs
        # print('Q_yearly prior',self.Q_yearly['prior'].mean())
        
    def load_Q_Naive(self):
        self.Q_Naive = pd.read_csv(join(self.OUTDIR,'Naive',f"{self.BASIN}_Q.csv"),
                                   index_col = 0, parse_dates = True)
    def load_Q_OSHD(self):
        Q_OSHD_list = []
        Q_obs_list = []
        for year in range(self.START_YEAR,self.END_YEAR+1):
            Q_OSHD = pd.read_csv(join(self.OUTDIR,'OSHD',f"{year}",f"{self.BASIN}_Q.csv"),
                                      index_col = 0, parse_dates = True)    
            Q_obs_list.append(Q_OSHD['obs'])
            new_cols = ['obs']+[col.split('_')[0] + '_' + col.split('_')[-1] for col in Q_OSHD.columns[1:]]
            Q_OSHD = Q_OSHD.rename(columns = dict(zip(Q_OSHD.columns,new_cols)))
            Q_OSHD_list.append(Q_OSHD)
        self.Q_OSHD = pd.concat(Q_OSHD_list)

        # self.Q_OSHD = pd.read_csv(join(self.OUTDIR,'OSHD',f"{self.BASIN}_Q.csv"),
        #                            index_col = 0, parse_dates = True)
    def load_Q_Synthetic(self):
        self.Q_Synthetic = pd.read_csv(join(self.SYNDIR,f"Q_{self.EXP_NAME}_{self.OBS_ERROR['scale']}.csv"),
                                   index_col = 0, parse_dates = True)
                    

    
    def calculate_OF(self, Q, index_cols):
        OF_df = pd.DataFrame(index=index_cols)
        pbias_df = pd.DataFrame(index=index_cols)
        self.Q_filtered = {}
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            setup = spot_setup(self.cfg, 
                               'placholdertje', 
                               calibration_purpose='Yearlycalib', 
                               soil_parset={}, 
                               single_year=year)
            # setup.OF = he.dr
            Qyear = Q.loc[f"{year-1}-10-01":f"{year}-09-30"]
            OF_df[str(year)] = np.zeros(len(index_cols))
            pbias_df[str(year)] = np.zeros(len(index_cols))
            self.Q_filtered[str(year)] = pd.DataFrame(index = Qyear.index)
            for i in OF_df.index:
                obs = Qyear['obs'].to_numpy()
                sim = Qyear[i].to_numpy()
                of,filtered ,pbias  = setup.objectivefunction(sim,obs,return_pbias=True,plot = False,
                                                     return_filtered=True)
                OF_df.loc[i, str(year)] = of
                pbias_df.loc[i,str(year)] = pbias 
                if not self.SWE_CALIB:
                    self.Q_filtered[str(year)].loc[filtered[0],i] = filtered[2]

                    if 'obs' not in self.Q_filtered[str(year)].columns:
                        self.Q_filtered[str(year)].loc[filtered[0],'obs'] = filtered[1]

        return OF_df, pbias_df

    def load_OFs(self):
        self.OFdic = {}
        self.PBIAS = {}
        # self.OF = 'kge'
        for key, Q in self.Q_yearly.items():
            if key != 'obs':
                self.OFdic[key],self.PBIAS[key] = self.calculate_OF(Q, Q.columns[1:])

        if self.SETTING == 'Real':
            self.OFdic['OSHD'], self.PBIAS['OSHD'] = self.calculate_OF(self.Q_OSHD, self.Q_OSHD.columns[1:])
            self.OFdic['Naive'],self.PBIAS['Naive'] = self.calculate_OF(self.Q_Naive, self.Q_Naive.columns[1:])
        # elif self.SETTING == 'Synthetic':
        #     self.OFdic['Synthetic'],self.PBIAS['Synthetic'] = self.calculate_OF(self.Q_Synthetic, self.Q_Synthetic.columns)
    
    # def calculate_MB(self):


    # def plot_OF(self):
    #     if 'OF' not in self.__dict__:
    #         self.load_OFs()
    #     # Check if the setting is 'Real'
    #     for OF in ['OF','PBIAS']:
    #         d = getattr(self,OF)
    #         # if self.SETTING == 'Real':
    #         # Create a list of dataframes for each method
    #         dfs = []
    #         if self.SETTING == 'Real':
    #             methods = ['prior', 'posterior']#, 'Naive', 'OSHD']
    #         elif self.SETTING == 'Synthetic':
    #             methods = ['prior','posterior']
    #         for method in methods:
    #             # Extract the data for the current method
    #             data = d[method]
    #             # Reshape the data to long format
    #             data = data.melt(var_name='Year', value_name='Objective Function')
    #             # Add the method as a column
    #             data['Method'] = method
    #             # Append the dataframe to the list
    #             dfs.append(data)
            
    #         # Concatenate the dataframes into a single dataframe
    #         df = pd.concat(dfs)
            
    #         # Plot the stripplot
    #         plt.figure(figsize=(12, 6))
    #         sns.stripplot(data=df, x='Year', y='Objective Function', hue='Method', palette=self.colors)
    #         plt.xlabel('Year')
    #         plt.grid(axis='y')
    #         plt.ylabel(OF)
    #         plt.title(f"{self.BASIN}_Objective Function for Different Methods")
    #         if OF == 'PBIAS':
    #             plt.axhline(0, color='black', linestyle='dashed')

    #         plt.savefig(join(self.FIGDIR,f"Yearly_{OF}s.png"), bbox_inches='tight', dpi=200)
    #             # plt.close()

    def plot_objective_function(self,objective_function, add_ax = False):
        if 'OFdic' not in self.__dict__:
            self.load_OFs()
        
        d = getattr(self, objective_function)
        dfs = []
        
        if self.SETTING == 'Real':
            methods = ['prior', 'posterior', 'Naive', 'OSHD']
        elif self.SETTING == 'Synthetic':
            methods = ['prior', 'posterior']
        
        for method in methods:
            data = d[method]
            data = data.melt(var_name='Year', value_name='Objective Function')
            data['Method'] = self.names[method]
            dfs.append(data)
        
        df = pd.concat(dfs)
        
        if add_ax ==False:
            f1,ax = plt.subplots(1,1,figsize=(12, 6))
        else:
            ax = add_ax
        sns.stripplot(ax = ax,data=df, x='Year', y='Objective Function', hue='Method', palette=self.colors2,
                      jitter = 0.15,dodge = False)
        ax.set_xlabel(None)
        ax.grid(axis='y')
        ax.set_ylabel(objective_function)
        ax.legend(title = None)
        if add_ax ==False:
            ax.set_title(f"{self.BASIN}_Objective Function for Different Methods")
        
        if objective_function == 'PBIAS':
            plt.axhline(0, color='black', linestyle='dashed')
        
        if add_ax == False:
            plt.savefig(join(self.FIGDIR, f"Yearly_{objective_function}s.png"), bbox_inches='tight', dpi=200)
        else:
            return ax
            
    def plot_snowmelt_OF(self, objective_function  = 'KGE', add_ax = False):
        synmelt = self.daily_scalars['snowmelt']['Synthetic']
        self.OF_snowmelt = {}
        for dist in ['prior','posterior']:
            OF = pd.DataFrame(index = self.Q_yearly[dist].columns[1:])
            snowmelt = self.daily_scalars['snowmelt'][dist]
            for col in OF.index:
                for year in range(self.START_YEAR,self.END_YEAR+1):
                    timeslice = slice(f"{year-1}-10-02",f"{year}-09-30")
                    if objective_function == 'KGE':
                        OF.loc[col,year] = spotpy.objectivefunctions.kge(snowmelt[col].loc[timeslice], synmelt.loc[timeslice])
            self.OF_snowmelt[dist] = OF
        

        dfs = []
        for dist in ['prior','posterior']:
            data = self.OF_snowmelt[dist]
            data = data.melt(var_name='Year', value_name='Objective Function')
            data['Method'] = self.names[dist]
            dfs.append(data)
        
        df = pd.concat(dfs)


        if add_ax == False:
            f1,ax = plt.subplots(1,1,figsize = (12,6))
        else:
            ax = add_ax
        # for dist in ['prior','posterior']:
        sns.stripplot(data = df, 
                        x = 'Year',
                        y = 'Objective Function', 
                        hue = 'Method',
                        ax = ax,
                        jitter = 0.15, 
                            palette = [self.colors['prior'],self.colors['posterior']],
                            )
        ax.grid(axis = 'y')
        ax.set_xlabel(None)
        ax.set_ylabel(f'Snowmelt {objective_function}')
        ax.legend()
        return ax

    def load_SWE(self):

        #yearly SWE             
        self.SWE = {}
        for dist in ['prior','posterior']:
            yearly_swe_list = []
            for year in range(self.START_YEAR,self.END_YEAR+1):
                swe_k = []
                for ksoil in range(self.SOIL_K):
                    snow =  xr.open_dataset(join(self.YEARLYDIR,
                                                                str(year),
                                                                str(ksoil),
                                                                f"{self.BASIN}_snow.nc"))
                    snowwater = xr.open_dataset(join(self.YEARLYDIR,
                                                                str(year),
                                                                str(ksoil),
                                                                f"{self.BASIN}_snowwater.nc"))
                    totalsnow = snowvars2swe(snow,snowwater)
                    swe_k.append(totalsnow)
                swe_all_k = xr.merge(swe_k)

                for var in swe_all_k.data_vars:
                    if dist not in var:
                        swe_all_k = swe_all_k.drop_vars(var)
                    else:
                        swe_all_k = swe_all_k.rename({var: f"{dist}_{var[-7:]}"})   
                yearly_swe_list.append(swe_all_k)
            self.SWE[dist] = xr.concat(yearly_swe_list, dim = 'time').sortby('lat',ascending=True).sortby('lon',ascending=True)

        if self.SETTING == 'Real':
            #OSHD
            C = SnowClass(self.BASIN)
            if self.OSHD == 'TI':
                OSHD = C.load_OSHD(root = join(self.ROOTDIR,'OSHD'),
                                   resolution = self.RESOLUTION)['swee_all'].rename(dict(x = 'lon', y = 'lat'))
            elif self.OSHD == 'EB':
                OSHD = C.load_FSM(root = join(self.ROOTDIR,'FSM'),
                                  resolution = self.RESOLUTION)['swet_all'].rename(dict(x = 'lon', y = 'lat'))
            OSHD = OSHD.sortby('lat',ascending=True).sortby('lon',ascending=True)
            OSHD = xr.where(np.isnan(self.dem),np.nan,OSHD)
            self.SWE['OSHD'] = OSHD

            #Naive
            snow = xr.open_dataset(join(self.OUTDIR,'Naive',f"{self.BASIN}_snow.nc"))
            snowwater = xr.open_dataset(join(self.OUTDIR,'Naive',f"{self.BASIN}_snowwater.nc"))
            totalsnow = snowvars2swe(snow,snowwater).to_array(name = 'Naive').squeeze().sortby('lat',ascending=True).sortby('lon',ascending=True)
            # totalsnow = xr.Dataset()

            # data_vars = snow.data_vars
            # first_var = list(data_vars.keys())[0][5:]
            # totalsnow['Naive'] = snow[f'snow_{first_var}'] + snowwater[f'snowwater_{first_var}']

            # totalsnow['Naive'] = snow["snow_Naive_ms0"] + snowwater['snowwater_Naive_ms0']
            self.SWE['Naive'] = totalsnow
        elif self.SETTING == 'Synthetic':
            SWEsyn = self.load_SWE_Synthetic()
            self.SWE['Synthetic'] = SWEsyn

    def load_SWE_Synthetic(self):
            if self.SYN_SNOWMODEL in ['seasonal','Hock','wflow']:
                # synswe = xr.open_dataset(join(self.SYNDIR,f"SWE_{self.EXP_NAME}.nc"))
                # synswe['SWE'] = synswe['snow'] + synswe['snowwater']
                # synswe = synswe.drop_vars(['snow','snowwater'])
                # synswe = synswe.drop_dims('layer')

                snow = xr.open_dataset(join(self.SYNDIR,f"{self.BASIN}_snow.nc"))
                snowwater = xr.open_dataset(join(self.SYNDIR,f"{self.BASIN}_snowwater.nc"))
                synswe = snowvars2swe(snow,snowwater).to_array(name = 'Synthetic').squeeze().sortby('lat',ascending=True).sortby('lon',ascending=True)
                # self.SWE['Synthetic'] = synswe
                return synswe
            #with OSHD
            elif self.SYN_SNOWMODEL in ['TI','EB','OSHD']:
                C = SnowClass(self.BASIN)
                if self.OSHD == 'TI':
                    OSHD = C.load_OSHD(root = join(self.ROOTDIR,'OSHD'),
                        resolution = self.RESOLUTION )['swee_all'].rename(dict(x = 'lon', y = 'lat'))
                    OSHD = OSHD.rename('SWE')
                elif self.OSHD == 'EB':
                    OSHD = C.load_FSM(root = join(self.ROOTDIR,'FSM'),
                                      resolution = self.RESOLUTION)['swet_all'].rename(dict(x = 'lon', y = 'lat'))
                    OSHD = OSHD.rename('SWE')
                OSHD = OSHD.sortby('lat',ascending=True).sortby('lon',ascending=True)
                OSHD = xr.where(np.isnan(self.dem),np.nan,OSHD)
                # self.SWE['Synthetic'] = OSHD
                return OSHD

    def load_SWEbands(self):
        self.SWEbands = {}

        #yearly SWE
        for dist in ['prior','posterior']:
            self.SWEbands[dist] = self.swe2bands(self.SWE[dist])

        if self.SETTING == 'Real':
            #OSHD
            self.SWEbands['OSHD'] = self.swe2bands(self.SWE['OSHD'])

            #Naive
            self.SWEbands['Naive'] = self.swe2bands(self.SWE['Naive'])
        elif self.SETTING == 'Synthetic':
            self.SWEbands['Synthetic'] = self.swe2bands(self.SWE['Synthetic'])
    def generate_3_bands(self):
        C= SnowClass(self.BASIN)
        dem = self.dem
        mini,maxi = dem.min().item(),dem.max().item()
        q25 = np.nanpercentile(dem,25).item()
        q75 = np.nanpercentile(dem,75).item()
        # bands = [np.round(mini,-2),np.round(dem.mean().item(),-2).item(),np.round(maxi,-2)]
        bands = [np.round(mini,-2),np.round(q25,-2),np.round(q75,-2),np.round(maxi,-2)]
        self.bands = bands 
        return bands
    def swe2bands(self,swe,bands = None):
        """Bands should be a list giving all boundaries, upper and lower included"""
        if (bands==None ) and (not hasattr(self,'bands')):
            bands = self.generate_3_bands()

        # res = int(self.RESOLUTION[:-1])
        C= SnowClass(self.BASIN)
        dem = self.dem
        swebands = {}
        for b in range(len(bands)-1):
            b0 = bands[b]
            b1 = bands[b+1]
            elevation_mask = xr.where((dem>=b0) & (dem<b1),1,0)

            # area = (np.sum(elevation_mask) * res*res).item()
            area = np.sum(elevation_mask ==1).item() * self.grid_area
            # print(area,' km2')
            sweband = swe*self.grid_area * 1e6 *0.001 # to m3
            sweband = sweband.where(elevation_mask ==1).sum(dim = ['lat','lon']).to_pandas()*(1/(area*1e6))*1000
            swebands[f"{b0}-{b1}"] = sweband

        # self.dem = dem
        self.bands = bands
        return swebands

    def composite_plot(self):
        bands = self.bands 
        if self.SETTING == 'Real':
            Naxes = len(bands) + 1
            height_ratios = [1]*(Naxes-1) + [2]
            labels = ['a)', 'b)', 'c)', 'd)','e)','f)','g)','h)'][:Naxes]
        elif self.SETTING == 'Synthetic':
            Naxes = len(bands) +2
            height_ratios = [1]*(Naxes-1) + [2]
            labels = ['a)', 'b)', 'c)', 'd)','e)','f)','g)','h)'][:Naxes]
        f, axes = plt.subplots(Naxes, 1, figsize=(12, 12),height_ratios=height_ratios)
        
        # Add labels for subplots
        pos = [0.03,0.95]
        fontsize = 14
        
        axes[0] = self.plot_objective_function('OFdic', add_ax=axes[0])
        axes[0].set_ylabel('Streamflow KGE')
        axes[0].text(pos[0],pos[1], labels[0], transform=axes[0].transAxes, fontsize=fontsize, va='top', ha='right')
        axes[0].set_ylim(0.2,1)

        if self.SETTING == 'Synthetic':
            axes[1] = self.plot_snowmelt_OF('KGE', add_ax=axes[1])
            # axes[1].set_ylabel('PBIAS')
            axes[1].text(pos[0],pos[1], labels[1], transform=axes[1].transAxes, fontsize=fontsize, va='top', ha='right')
        
        for i in range(len(bands) - 1):
            b0 = bands[i]
            b1 = bands[i + 1]
            ax = self.plot_sweband(b0, b1, ax=axes[i + 1 + 1*int(self.SETTING == 'Synthetic')])
            ax.text(pos[0],pos[1], labels[i + 1], transform=ax.transAxes, fontsize=fontsize,  va='top', ha='right')
            if i == 0:
                ax.legend()
            else:
                ax.legend().remove()
        
        axes[-1] = self.plot_waterbalance(add_ax=axes[-1])
        axes[-1].text(pos[0],pos[1], labels[-1], transform=axes[-1].transAxes, fontsize=fontsize,  va='top', ha='right')

        if self.SETTING == 'Real':
            suptitle = f"{self.BASIN} - Real-world case study"
        elif self.SETTING == 'Synthetic':
            suptitle = f"{self.BASIN} - Idealized case study"

        f.suptitle(suptitle, fontsize=20)
        plt.tight_layout()
        plt.savefig(join(self.FIGDIR, f"{self.BASIN}_composite_{self.SETTING}.png"), bbox_inches='tight', dpi=300)

    def plot_swebands(self):
        bands = self.bands 
        f,axes = plt.subplots(len(bands)-1,1,figsize = (20,10))
        for i in range(len(bands)-1):
            b0 = bands[i]
            b1 = bands[i+1]
            ax = self.plot_sweband(b0,b1,ax = axes[i])
            ax.set_xlabel(None)
        plt.tight_layout()
        plt.suptitle(f"{self.BASIN} - SWE bands", fontsize = 20, y= 1.1)
        plt.savefig(join(self.FIGDIR,f"{self.BASIN}_swebands.png"), bbox_inches='tight', dpi=200)
        

    def plot_sweband(self,b0,b1, ax = None):
        timeslice = slice(f"{self.START_YEAR-1}-10-01",f"{self.END_YEAR}-09-30")
        for dist in ['prior','posterior']:
            labelflag = False
            for col in self.SWEbands[dist][f"{b0}-{b1}"].columns:
                if col =='spatial_ref':
                    continue
                if labelflag == False:
                    labelflag = True
                    label = self.names[dist]
                else:
                    label = '_noLegend'
                self.SWEbands[dist][f"{b0}-{b1}"].loc[timeslice][col].plot(ax = ax,
                                                                               label = label,
                                                                               color = self.colors[dist],
                                                                               alpha = self.alphas[dist])
        if self.SETTING == 'Real':
            self.SWEbands['OSHD'][f"{b0}-{b1}"].loc[timeslice].plot(ax = ax, 
                                                                    label = f'Reference ({self.names["OSHD"]})',
                                                                    color = self.colors['OSHD'],)

            labelflag = False
            for col in self.SWEbands['Naive'][f"{b0}-{b1}"].columns:
                if col =='spatial_ref':
                    continue
                if labelflag == False:
                    labelflag = True
                    label = self.names['Naive']
                else:
                    label = '_noLegend'
                self.SWEbands['Naive'][f"{b0}-{b1}"].loc[timeslice][col].plot(ax = ax, 
                                                                              label = label
                                                                              , color = self.colors['Naive'])
        elif self.SETTING == 'Synthetic':
            self.SWEbands['Synthetic'][f"{b0}-{b1}"].loc[timeslice].plot(ax = ax,
                                                                            label = 'Reference (OSHD)',
                                                                            color = self.colors['Synthetic'])



        ax.set_title(f"{int(b0)}-{int(b1)} m")
        ax.set_ylabel('SWE [mm]')
        ax.set_xlabel(None)
        ax.legend()
        ax.grid()
        return ax

    def load_stations(self):
        C = SnowClass(self.BASIN)
        C.load_HS_obs()
        C.load_SLF_SWE()
        C.calculated_benchmarks(include_unreduced_benchmark = True)
        self.C = C 
        

        MODSWE = self.SWE['posterior'].mean(dim=['lat', 'lon']).to_pandas()

        self.normed_SWE = {}
        self.normed_SWE['posterior'] = normalize_swe(MODSWE)
        self.normed_SWE['OSHD'] = normalize_swe(self.SWE['OSHD'].mean(dim = ['lat','lon']).to_dataframe()).rename(
            columns = {'swee_all':'OSHD'}
        )
        self.normed_SWE['Naive'] = normalize_swe(self.SWE['Naive'].mean(dim = ['lat','lon']).to_dataframe())

        self.normed_SWE_m = {}
        self.normed_SWE_m['posterior'] = self.normed_SWE['posterior'].resample('M').mean()
        self.normed_SWE_m['OSHD'] = self.normed_SWE['OSHD'].resample('M').mean()
        self.normed_SWE_m['Naive'] = self.normed_SWE['Naive'].resample('M').mean()
        
        
        self.normed_SWE_station = {}
        self.normed_SWE_station['posterior'] = {}
        self.normed_SWE_station['OSHD'] = {}
        self.normed_SWE_station['Naive'] = {}
        for ii,station in enumerate(self.C.station_short_names):
                        # print(station)
            SWE = self.C.HSSWE[station]
            SWE['normalized'] = SWE.apply(
                        lambda row: row['SWE'] / row['SWE_DOYmean'] if row['SWE'] > 10 and row['SWE_DOYmean'] > 10 else None,
                        axis=1
                    )
            self.normed_SWE[station] = SWE['normalized']
            self.normed_SWE_m[station] = SWE['normalized'].resample('M').mean()


            elevation = self.C.station_elevation[station]
            bandwidth = 300
            masked_elevation = self.dem.where(
                (self.dem >= elevation - 0.5*bandwidth) & (self.dem <= elevation + 0.5* bandwidth), np.nan)
            if np.sum(masked_elevation.notnull()) == 0:
                print(f"Station {station} is not in the DEM")
                masked_elevation = self.dem.where(
                    self.dem < self.dem+ 0.5*bandwidth, np.nan)
            masked_elevation = masked_elevation.drop_vars('spatial_ref')
            masked_SWE = self.SWE['posterior'].where(masked_elevation.notnull()).mean(dim = ['lat','lon']).to_pandas()
            self.normed_SWE_station['posterior'][station] = normalize_swe(masked_SWE).resample('M').mean()
            masked_SWE = self.SWE['OSHD'].where(masked_elevation.notnull()).mean(dim = ['lat','lon']).to_dataframe()
            self.normed_SWE_station['OSHD'][station] = normalize_swe(masked_SWE).resample('M').mean()
            masked_SWE = self.SWE['Naive'].where(masked_elevation.notnull()).mean(dim = ['lat','lon']).to_dataframe()
            self.normed_SWE_station['Naive'][station] = normalize_swe(masked_SWE).resample('M').mean()

    def stations_OF(self):
        if not hasattr(self,'C'):
            self.load_stations()
        OF = {}
        methods = ['posterior','OSHD','Naive']
        timeslice = self.normed_SWE_m['posterior'].index
        #keep only months from november to may
        timeslice = timeslice[np.isin(timeslice.month,[11,12,1,2,3,4,5])]
        # timeslice = pd.date_range(start = f"{year-1}-11-01", end = f"{year}-05-30", freq = 'M')

        for station in self.C.station_short_names:
            OF[station] = pd.DataFrame(columns = [self.names[m] for m in methods],
                    index =np.arange(len(timeslice)*self.SOIL_K*self.YEARLY_K ))
            for m in methods:
                name= self.names[m]
                values = []
                for col in self.normed_SWE_station[m][station].columns:
                    obs = self.normed_SWE_m[station][timeslice]
                    sim = self.normed_SWE_station[m][station][col][timeslice]
                # for col in self.normed_SWE_m[m].columns:
                #     obs = self.normed_SWE_m[station][timeslice]
                #     sim = self.normed_SWE_m[m][col][timeslice]
                    # logdif = np.abs(np.log(obs)-np.log(sim)).values
                    logratio = np.abs(np.log(obs/sim)).values
                    values.append(logratio)
                allvalues = np.concatenate(values)
                OF[station].loc[np.arange(len(allvalues)),name] =allvalues
            OF[station]['station'] = station
            OF[station] = OF[station].melt(id_vars = 'station',var_name = 'Method', value_name = 'Objective Function')
        
        melted = pd.concat([OF[station] for station in self.C.station_short_names])
        melted = melted[~melted['Objective Function'].isna()]
        melted.index = np.arange(len(melted))

        plt.figure()
        sns.boxplot(data = melted, x = 'station', 
        y = 'Objective Function', hue = 'Method',palette=self.colors2,
        dodge = True,gap = 0.1)
        plt.legend(title = None)
        plt.grid(axis = 'y')
        # plt.ylabel(r"$| \log(\text{obs}_{\text{norm}}) - \log(\text{sim}_{\text{norm}}) |$")
# print(formula))
        # plt.axhline(0, color = 'black', linestyle = 'dashed')
        plt.ylabel("Absolute log-ratio normSWE")
        plt.title(f"{self.BASIN} - Station comparison")
        plt.ylim(0,5)
        # plt.semilogy()
        plt.savefig(join(self.FIGDIR,f"{self.BASIN}_Station_comparison_abslog.png"), bbox_inches='tight', dpi=200)
                # OF[station].append({'m':values})

            # OF[station][dist] = self.calculate_OF(self.Q_yearly[dist], [station])[0]
    
    def station_SWEmax(self):
        if not hasattr(self,'C'):
            self.load_stations()
        
        C = self.C
        station_swemax = {}
        daterange = pd.date_range(start = f"{self.START_YEAR-1}-10-01", end = f"{self.END_YEAR}-09-30", freq = 'D')
        # Take the SWEmax of each year of each station
        for station in C.station_short_names:
            station_swemax[station] = {}
            station_swe = C.HSSWE[station]['SWE'].sort_index().loc[slice(f"{self.START_YEAR}",f"{self.END_YEAR}")]
            station_swemax[station][station] = station_swe.resample('Y').max()

            elevation = self.C.station_elevation[station]
            bandwidth = 300
            masked_elevation = self.dem.where(
                (self.dem >= elevation - 0.5*bandwidth) & (self.dem <= elevation + 0.5* bandwidth), np.nan)
            if np.sum(masked_elevation.notnull()) == 0:
                print(f"Station {station} is not in the DEM")
                masked_elevation = self.dem.where(
                    self.dem < self.dem+ 0.5*bandwidth, np.nan)
            masked_elevation = masked_elevation.drop_vars('spatial_ref')

            
            for dist in ['posterior','OSHD','Naive']:
                masked_SWE = self.SWE[dist].where(masked_elevation.notnull()).mean(dim = ['lat','lon']).to_pandas()
                masked_SWEmax = masked_SWE.resample('Y').max()
                station_swemax[station][dist] = masked_SWEmax
        #savefig
        from scipy.stats import linregress

        # Assuming C, station_swemax, self.names, self.colors, self.FIGDIR, and self.BASIN are defined

        f1, axes = plt.subplots(1, len(C.station_short_names), figsize=(5 * len(C.station_short_names), 5))
        for ii, station in enumerate(C.station_short_names):
            years = station_swemax[station][station].index
            ax = axes[ii]
            for dist in ['posterior', 'OSHD', 'Naive']:
                if dist == 'posterior':
                    x = station_swemax[station][station]

                    # Compute linear regression
                    ymedian = station_swemax[station][dist].median(axis=1).loc[years]
                    slope, intercept, r_value, p_value, std_err = linregress(x, ymedian)
                    ax.plot(x, slope * x + intercept, color=self.colors[dist], linestyle='dotted')
                    # ax.text(0.05, 0.95, f'R²={r_value**2:.2f}', transform=ax.transAxes,
                            #  fontsize=8, verticalalignment='top')
                    for col in station_swemax[station][dist].columns:
                        if col == 'spatial_ref':
                            continue
                        if 'ms0_my0' in col:
                            label = f"{self.names[dist]} \n R²={r_value**2:.2f}"
                        else:
                            label = '_noLegend'
                        y = station_swemax[station][dist][col].loc[years]
                        ax.scatter(x, y, label=label, color=self.colors[dist])
                else:
                    x = station_swemax[station][station]
                    y = station_swemax[station][dist].loc[years]

                    if dist == 'Naive':
                        y = y['Naive']
                    
                    # Compute linear regression
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    ax.plot(x, slope * x + intercept, color=self.colors[dist], linestyle='dotted')
                    # ax.text(0.05, 0.95, f'R²={r_value**2:.2f}', transform=ax.transAxes, fontsize=8, verticalalignment='top')
                    
                    label = f"{self.names[dist]} \n R²={r_value**2:.2f}"
                    ax.scatter(x, y, label=label, color=self.colors[dist])

            axmin = np.min([ax.get_xlim()[0], ax.get_ylim()[0]])
            axmax = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])

            ax.set_xlim(axmin, axmax)
            ax.set_ylim(axmin, axmax)
            ax.legend()
            ax.grid()
            ax.plot(np.arange(axmin, axmax), np.arange(axmin, axmax), color='black', linestyle='dashed')
            ax.set_title(f"{station} at {C.station_elevation[station]} m elevation")
            ax.set_xlabel('Station SWEmax [mm]')
        axes[0].set_ylabel('Model SWEmax [mm]')
        f1.savefig(join(self.FIGDIR, f"{self.BASIN}_Station_SWEmax_scatterplot.png"), bbox_inches='tight', dpi=200)
        # f1,axes = plt.subplots(1,len(C.station_short_names),
        #                        figsize = (4*len(C.station_short_names),4))
        # for ii,station in enumerate(C.station_short_names):
        #     years = station_swemax[station][station].index
        #     ax = axes[ii]
        #     for dist in ['posterior','OSHD','Naive']:
        #         if dist == 'posterior':
        #             for col in station_swemax[station][dist].columns:
        #                 if col == 'spatial_ref':
        #                     continue
        #                 if 'ms0_my0' in col:
        #                     label = self.names[dist]
        #                 else:
        #                     label = '_noLegend'
        #                 ax.scatter(station_swemax[station][station],
        #                            station_swemax[station][dist][col].loc[years],
        #                            label = label,
        #                            color = self.colors[dist])
        #         else:
        #             ax.scatter(station_swemax[station][station],
        #                        station_swemax[station][dist].loc[years],
        #                        label = self.names[dist],
        #                        color = self.colors[dist])
        #     axmin = np.min([ax.get_xlim()[0],ax.get_ylim()[0]])
        #     axmax = np.max([ax.get_xlim()[1],ax.get_ylim()[1]])

        #     ax.set_xlim(axmin,axmax)
        #     ax.set_ylim(axmin,axmax)
        #     ax.legend()
        #     ax.grid()
        #     ax.plot(np.arange(axmin,axmax),np.arange(axmin,axmax),color = 'black',linestyle = 'dashed')
        #     ax.set_title(f"{station} at {C.station_elevation[station]} m elevation")
        #     ax.set_xlabel('Station SWEmax [mm]')
        # axes[0].set_ylabel('Model SWEmax [mm]')
        # f1.savefig(join(self.FIGDIR,f"{self.BASIN}_Station_SWEmax_scatterplot.png"), bbox_inches='tight', dpi=200)

        
        f1,axes = plt.subplots(1,len(C.station_short_names),
                               figsize = (5*len(C.station_short_names),5))
        for ii,station in enumerate(C.station_short_names):
            
            years = station_swemax[station][station].index
            ax = axes[ii]
            for dist in ['posterior','OSHD','Naive']:
                if dist == 'posterior':
                    for col in station_swemax[station][dist].columns:
                        if col == 'spatial_ref':
                            continue
                        if 'ms0_my0' in col:
                            label = self.names[dist]
                        else:
                            label = '_noLegend'
                        station_swemax[station][dist][col].loc[years].plot(
                            ax = ax,   label = label,
                                   color = self.colors[dist])
                else:
                    station_swemax[station][dist].loc[years].plot(
                        ax = ax,
                               label = self.names[dist],
                               color = self.colors[dist])
            station_swemax[station][station].plot(ax = axes[ii],
                                                  label = 'Station',
                                                  color = 'black',
                                                  ls = 'dashed'
                                                  )
            ax.legend()
            ax.grid()
            ax.set_title(f"{station} at {C.station_elevation[station]} m elevation")
            # ax.set_xlabel('Station SWEmax [mm]')
        axes[0].set_ylabel('SWEmax [mm]')
        f1.savefig(join(self.FIGDIR,f"{self.BASIN}_Station_SWEmax_timeseries.png"), bbox_inches='tight', dpi=200)
            
                # ax.scatter(station_swemax[station][station],station_swemax[station][dist],label = self.names[dist])
                # ssm = pd.DataFrame(columns = self.SWE[dist].data_vars,
                #                                 index = np.arange(self.START_YEAR,self.END_YEAR+1))
                # for col in ssm.columns:
                #     ssm[col] = self.SWE[dist][col].loc[daterange].resample('Y').max().values
                # sim_swemax[dist]

        #Take the SWEmax of each year of each simulation


        #plot against each other
        
    def plot_each_station(self, add_ax = False):

        if not hasattr(self,'C'):
            self.load_stations()
        Naxesx = self.END_YEAR-self.START_YEAR+1
        f1,axes = plt.subplots(len(self.C.station_short_names),Naxesx,figsize = (15,9),sharex = 'col',sharey = 'row')
        # axes = axes.flatten()
        plt.subplots_adjust(wspace = 0.05)
        for ii,station in enumerate(self.C.station_short_names):
            # subaxes = axes[ii:Naxesx+ii]
            # if add_ax == False:
            # else:
            #     ax = add_ax

            for i,year in enumerate(range(self.START_YEAR,self.END_YEAR+1)):
                ax = axes[ii,i]
                daterange = pd.date_range(start = f"{year-1}-11-01", end = f"{year}-05-30", freq = 'M')
                label_flag = False
                for col in self.normed_SWE_station['posterior'][station].columns:
                    if label_flag == False:
                        label = self.names['posterior']
                        label_flag = True
                    else:
                        label = '_noLegend'
                    self.normed_SWE_station['posterior'][station][col].loc[daterange].plot(ax = ax, color = 'tab:blue', 
                    label = label, alpha = 0.5)
                # SWE_normed.loc[daterange].plot(ax = ax, color = 'tab:blue', label = 'Posteriors', alpha = 0.7)
                self.normed_SWE_station['OSHD'][station].loc[daterange].rename(columns={'swee_all':self.names['OSHD']}).plot(ax = ax, color = 'tab:red', 
                    label = self.names['OSHD'], alpha = 0.9)
                self.normed_SWE_station['Naive'][station].loc[daterange].rename(columns={'Naive':self.names['Naive']}).plot(ax = ax, color = 'saddlebrown',
                 label = self.names['Naive'], alpha = 0.9)

                self.normed_SWE_m[station].reindex(daterange).plot(ax=ax, label="Station " + station, color='black',
                    linestyle='None', marker='x', markersize = 5, markerfacecolor = 'None' )
                ax.grid()
                ax.axhline(1, color = 'black', linestyle = 'dashed')
                if i ==3:          
                    ax.legend()

                    ax.set_title(f"{station} at {self.C.station_elevation[station]} m elevation")
                else:
                    ax.legend().remove()
                # ax.semilogy()
                # ax.set_ylim(0.2,5)
                if (i == 0) & (ii == 1):
                    ax.set_ylabel('Normalized SWE [-]')
                
            # axes[0].legend(ncol = 2,loc = [0.1,1.1])
            # axes[0].set_ylabel('Normalized SWE [-]')

            f1.suptitle(f"{self.BASIN} - Normalized SWE", fontsize = 15, y =0.98)
            plt.savefig(join(self.FIGDIR,f"{self.BASIN}_Normalized_SWE_stations.png"), bbox_inches='tight', dpi=200)




    def plot_stations(self):
        if not hasattr(self,'C'):
            self.load_stations()
        

        # Example usage
        
        f1,axes = plt.subplots(1,self.END_YEAR-self.START_YEAR+1,figsize = (15,3),sharey = True)
        plt.subplots_adjust(wspace = 0.05)
        markers = ['o','x','^','v','<','>','D','P','X','*']
        for i,year in enumerate(range(self.START_YEAR,self.END_YEAR+1)):
            ax = axes[i]
            daterange = pd.date_range(start = f"{year-1}-11-01", end = f"{year}-05-30", freq = 'M')
            label_flag = False
            for col in self.normed_SWE_m['posterior'].columns:
                if label_flag == False:
                    label = self.names['posterior']
                    label_flag = True
                else:
                    label = '_noLegend'
                self.normed_SWE_m['posterior'][col].loc[daterange].plot(ax = ax, color = 'tab:blue', label = label, alpha = 0.5)
            # SWE_normed.loc[daterange].plot(ax = ax, color = 'tab:blue', label = 'Posteriors', alpha = 0.7)
            self.normed_SWE_m['OSHD'].loc[daterange].plot(ax = ax, color = 'tab:red', label = self.names['OSHD'], alpha = 0.9)
            self.normed_SWE_m['Naive'].loc[daterange].plot(ax = ax, color = 'saddlebrown', label = self.names['Naive'], alpha = 0.9)

            for ii,station in enumerate(self.C.station_short_names):
                self.normed_SWE_m[station].reindex(daterange).plot(ax=ax, label="Station " + station, color='black',
                 linestyle='None', marker=markers[ii], markersize = 5, markerfacecolor = 'None' )
            ax.grid()
            ax.axhline(1, color = 'black', linestyle = 'dashed')
            # ax.semilogy()
            # ax.set_ylim(0.2,5)
            # if i == 0:
            #     ax.legend(ncols = 2)
            #     ax.set_ylabel('Normalized SWE [-]')
        # else:
            ax.legend().remove()
        axes[0].legend(ncol = 2,loc = [0.1,1.1])
        axes[0].set_ylabel('Normalized SWE [-]')
        f1.suptitle(f"{self.BASIN} - Normalized SWE", fontsize = 15, y = 1)
        plt.savefig(join(self.FIGDIR,f"{self.BASIN}_Normalized_SWE_m.png"), bbox_inches='tight', dpi=200)

        
        
        # daterange = pd.date_range(start = f"{self.START_YEAR-1}-10-01", end = f"{self.END_YEAR}-09-30", freq = 'D')
        # daterange = daterange[np.isin(daterange.month,[11,12,1,2,3,4,5,6])]   



        # f1,axes = plt.subplots(1,self.END_YEAR-self.START_YEAR+1,figsize = (20,5),sharey = True)
        # plt.subplots_adjust(wspace = 0.05)
        # for i,year in enumerate(range(self.START_YEAR,self.END_YEAR+1)):
        #     ax = axes[i]
        #     daterange = pd.date_range(start = f"{year-1}-11-01", end = f"{year}-05-30", freq = 'D')
        #     label_flag = False
        #     for col in SWE_normed.columns:
        #         if label_flag == False:
        #             label = self.names['posterior']
        #             label_flag = True
        #         else:
        #             label = '_noLegend'
        #         SWE_normed[col].loc[daterange].plot(ax = ax, color = 'tab:blue', label = label, alpha = 0.5)
        #     # SWE_normed.loc[daterange].plot(ax = ax, color = 'tab:blue', label = 'Posteriors', alpha = 0.7)
        #     OSHD_normed.loc[daterange].plot(ax = ax, color = 'tab:red', label = self.names['OSHD'], alpha = 0.9)
        #     Naive_normed.loc[daterange].plot(ax = ax, color = 'saddlebrown', label = self.names['Naive'], alpha = 0.9)

        #     for station in self.C.station_short_names:
        #         # print(station)
        #         SWE = self.C.HSSWE[station]
        #         SWE['normalized'] = SWE.apply(
        #                     lambda row: row['SWE'] / row['SWE_DOYmean'] if row['SWE'] > 10 and row['SWE_DOYmean'] > 10 else None,
        #                     axis=1
        #                 )
        #         try:
        #             SWE['normalized'].loc[daterange].plot(ax = ax, label = "Station "+station, color = 'black',linestyle = 'dashed')
        #         except:
        #             # print(station)
        #             SWE['normalized'].reindex(daterange).plot(ax = ax, label = "Station "+station, color = 'black',linestyle = 'dashed')
        #     ax.grid()
        #     ax.set_ylim(0,4)
        #     if i == 0:
        #         ax.legend()
        #         ax.set_ylabel('Normalized SWE [-]')
        #     else:
        #         ax.legend().remove()
        # f1.suptitle(f"{self.BASIN} - Normalized SWE", fontsize = 15, y = 1)
        # plt.savefig(join(self.FIGDIR,f"{self.BASIN}_Normalized_SWE.png"), bbox_inches='tight', dpi=200)
        # # daterange = pd.date_range(start = f"{self.START_YEAR-1}-10-01", end = f"{self.END_YEAR}-09-30", freq = 'D')
        # # daterange = daterange[np.isin(daterange.month,[11,12,1,2,3,4,5,6])]   

    def load_pars(self):
        #we want to load the pars from the soilcalib and from yearly calib
        #Lets start with just the ksoil best pars and then the kyearly best pars 
        #the kyearly should be one dataframe per parameter, with the 25 kyearly as index and the years as columsn 

        self.pars = {}

        #Ksoil
        self.pars['Soilpars'] = {}
        for dist in ['prior','posterior']:
            FOLDER = join(self.OUTDIR,f"Soilcalib")
            FILE = join(FOLDER, f"{self.BASIN}_ksoil{self.SOIL_K}_{dist}.csv")
            self.pars['Soilpars'][dist] = pd.read_csv(FILE,header=0, index_col=0)
            # if not 'rfcf' in self.pars['Soilpars'][dist].columns:
            #     self.pars['Soilpars'][dist]['rfcf'] = self.param_fixed['rfcf']

        #Kyearly
        names = [f"ms{ksoil}_my{kyearly}" for ksoil in range(self.SOIL_K) for kyearly in range(self.YEARLY_K)]
        self.pars['Yearlypars'] = {}
        for par,value in self.param_ranges.items():
                if self.param_kinds[par] in ['snow','meteo']:
                    self.pars['Yearlypars'][par] = {}
                    for dist in ['prior','posterior']:
                        self.pars['Yearlypars'][par][dist] = pd.DataFrame(index = names, columns = range(self.START_YEAR,self.END_YEAR+1))



        for yy,year in enumerate(range(self.START_YEAR,self.END_YEAR+1)):
            counter = 0
            for ksoil in range(self.SOIL_K):
                for dist in ['prior','posterior']:
                    FOLDER = join(self.OUTDIR,f"Yearlycalib",f"{year}",f"{ksoil}")
                    FILE = join(FOLDER, f"{self.BASIN}_ms{ksoil}_kyearly{self.YEARLY_K}_{dist}.csv")
                    kyearlypost = pd.read_csv(FILE,header=0, index_col=0)
                    # names = [f"ms{ksoil}_my{kyearly}" for kyearly in range(self.YEARLY_K)]

                    for par in self.pars['Yearlypars']:
                        self.pars['Yearlypars'][par][dist].iloc[counter:counter+self.YEARLY_K,yy] = kyearlypost[par].values
                counter += self.YEARLY_K
        
        # if not 'rfcf' in self.pars['Yearlypars'].keys():
        #     self.pars['Yearlypars']['rfcf'] = {}
        #     for dist in ['prior','posterior']:
        #         self.pars['Yearlypars']['rfcf'][dist] = pd.DataFrame(index = np.arange(self.SOIL_K),columns = range(self.START_YEAR,self.END_YEAR+1))
        #         self.pars['Yearlypars']['rfcf'][dist].iloc[:,:] = self.param_fixed['rfcf']
        #OB pars
        if self.SETTING == 'Real':
            OB_sfcf = pd.DataFrame(index = np.arange(self.SOIL_K),columns = range(self.START_YEAR,self.END_YEAR+1))
            for ksoil in range(self.SOIL_K):
                for yy,year in enumerate(range(self.START_YEAR,self.END_YEAR+1)):
                    FOLDER = join(self.OUTDIR,'OSHD',f"{year}",f"{ksoil}")
                    FILE = join(FOLDER, f"{self.BASIN}_{year}_ms{ksoil}_OB_calib.csv")
                    rfcf = pd.read_csv(FILE,header=0)
                    #find the value in parrfcf column that correponds to the highest value in the like1 column
                    rfcf_value = rfcf.loc[rfcf['like1'].idxmax()]['parrfcf']
                    OB_sfcf.loc[ksoil,year] = rfcf_value
            self.pars['Yearlypars']['rfcf']['OSHD'] = OB_sfcf

    def plot_pars(self):
        if 'pars' not in self.__dict__:
            self.load_pars()
        self.plot_soilpars()
        self.plot_yearlypars()

    def plot_yearlypars(self):
        if 'pars' not in self.__dict__:
            self.load_pars()

        fig, axes = plt.subplots(len(self.pars['Yearlypars']), 
                                 1, 
                                 figsize=(10, 2 * len(self.pars['Yearlypars'])), 
                                 sharex=True)

        for i, (par, values) in enumerate(self.pars['Yearlypars'].items()):
            ax = axes[i]
            for dist, df in values.items():
                sns.stripplot(data=df, 
                              ax=ax, 
                              color=self.colors[dist], 
                              dodge=False, 
                              legend=False,
                                  size=4, jitter=0.2, linewidth=0.2)
                sns.violinplot(data=df, ax = ax, color = self.colors[dist], alpha = self.alphas[dist])
                # df.plot(ax=ax, label=dist, alpha=0.5, color = self.colors[dist])
            ax.set_ylim(self.param_ranges[par])
            # ax.set_title(par)
            ax.set_ylabel(par)
            # print()
            # for ii in self.pars['Soilpars'].index:
            #     ax.axhline(self.pars['Soilpars'].loc[ii,par], color = 'black', linestyle = 'dashed')
            if self.SETTING == 'Synthetic' and self.SYN_SNOWMODEL in ['seasonal','Hock'] and par =='sfcf':
                for year in range(self.START_YEAR,self.END_YEAR+1): 
                    ax.hlines(self.trueparams[f"sfcf_{year}"], 
                              xmin = year-self.START_YEAR-0.5, xmax = year-self.START_YEAR+0.5, 
                              color = 'black', linestyle = 'dashed')
                           
                
                
                # for year in range(self.START_YEAR,self.END_YEAR+1):
                #     ax.hlines(self.trueparams[par][f"sfcf_{year}"], xmin = year-0.5, xmax = year+0.5, color = 'black', linestyle = 'dashed')
                
                # ax.hlines(self.trueparams[par], color = 'black', linestyle = 'dashed')

        title_text = (
            f"<size:18>{self.BASIN} - <color:{self.colors['prior']}>Prior</> and <color:{self.colors['posterior']}>posterior</> yearly parameters</>"
            )
        flexitext(0.5, 1, title_text, va="bottom", ha="center", xycoords="figure fraction")

            # ax.set_title("Prior and posterior parameter ensemble members", color=self.colors['prior'] + self.colors['posterior'])
            # ax.legend()

        plt.tight_layout()
        plt.savefig(join(self.FIGDIR,f"{self.BASIN}_Yearly_pars.png"), bbox_inches='tight', dpi=200)
        # plt.close()

    def plot_soilpars(self):
        if 'pars' not in self.__dict__:
            self.load_pars()
        soilpars = [par for par in self.pars['Soilpars']['prior'].columns if self.param_soilcalib[par] == True]
        fig, axes = plt.subplots(1, len(soilpars), figsize=(5,2))
        for i,par in enumerate(soilpars):
            if self.param_soilcalib[par] == True:
                for dist in ['prior','posterior']:
                    df = self.pars['Soilpars'][dist][par]
                    sns.stripplot(data=df, ax=axes[i], color=self.colors[dist], dodge=False, legend=False, size=4, jitter=0.2, linewidth=0.2)
                if self.SETTING == 'Synthetic':
                    if not par == 'sfcf':
                        axes[i].axhline(self.trueparams[par], color = 'black', linestyle = 'dashed')
                    

        title_text = (
            f"<size:18>{self.BASIN} - <color:{self.colors['prior']}>Prior</> and <color:{self.colors['posterior']}>posterior</> soil parameters</>"
            )
        flexitext(0.5, 1, title_text, va="bottom", ha="center", xycoords="figure fraction")

        plt.tight_layout()
        plt.savefig(join(self.FIGDIR,f"{self.BASIN}_Soil_pars.png"), bbox_inches='tight', dpi=200)
        # plt.close()        
        # 

     
    def load_scalaroutput(self):
        def process_data(file_path, sfcf, rfcf, start_date, end_date, SWE_source=None):
            df = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True).loc[start_date:end_date]
            if SWE_source is not None:
                df['SWE'] = SWE_source
            else:
                df['SWE'] = df['snow'] + df['snowwater']
            diff = df['SWE'].diff().fillna(0)
            df['snowmelt'] = diff.where(diff < 0, 0)*-1
            df['snowfall'] = diff.where(diff > 0, 0)
            # df['snowfall'] = df['SWE'].diff().fillna(0)
            # df['snowfall'][df['snowfall'] > 0] = 0
            # df['snowmelt'] = df['SWE'].diff().fillna(0)
            # df['snowmelt'][df['snowmelt'] < 0] = 0
            df['snowfall_orig'] = df['snowfall'] / sfcf
            df['rainfall_orig'] = df['Pmean'] - df['snowfall_orig']
            df['rainfall'] = df['rainfall_orig'] * rfcf
            df['Ptotal'] = df['rainfall'] + df['snowfall']
            df['Q'] *= 3600 * 24 * (1 / (self.dem_area * 1e6)) * 1000
            return df

        self.scalars = {}
        for dist in ['prior', 'posterior']:
            self.scalars[dist] = {}
            for ksoil in range(self.SOIL_K):
                for kyearly in range(self.YEARLY_K):
                    df_list = []
                    for year in range(self.START_YEAR, self.END_YEAR + 1):
                        FOLDER = join(self.OUTDIR, "Yearlycalib", f"{year}", f"{ksoil}")
                        FILE = join(FOLDER, f"{self.BASIN}_Yearlycalib_{self.EXP_ID}_{self.BASIN}_{year}_{dist}_ms{ksoil}_my{kyearly}.csv")
                        sfcf = self.pars['Yearlypars']['sfcf'][dist].loc[f"ms{ksoil}_my{kyearly}", year]
                        try:
                            rfcf = self.pars['Yearlypars']['rfcf'][dist].loc[f"ms{ksoil}_my{kyearly}", year]
                        except:
                            rfcf = self.param_fixed['rfcf']
                        yearlydf = process_data(FILE, sfcf, rfcf, f"{year-1}-10-01", f"{year}-09-30")
                        df_list.append(yearlydf)
                    df = pd.concat(df_list)
                    self.scalars[dist][f"ms{ksoil}_my{kyearly}"] = df
            self.scalar_names = df.columns

            self.scalars[dist]['Soilcalib'] = {}
            for ksoil in range(self.SOIL_K):
                FILE = join(self.OUTDIR, 'Soilcalib', f"{self.BASIN}_Soilcalib_{self.EXP_ID}_{self.BASIN}_{dist}_ms{ksoil}.csv")
                sfcf = self.pars['Soilpars'][dist].loc[ksoil, 'sfcf']
                try:
                    rfcf = self.pars['Soilpars'][dist].loc[ksoil, 'rfcf']
                except:
                    rfcf = self.param_fixed['rfcf']
                df = process_data(FILE, sfcf, rfcf, f"{self.START_YEAR-1}-10-01", f"{self.END_YEAR}-09-30")
                self.scalars[dist]['Soilcalib'][f"ms{ksoil}"] = df

        if self.SETTING == 'Real':
            self.scalars['OSHD'] = {}
            # for ksoil in range(self.SOIL_K):
            #     FILE = join(self.OUTDIR, 'OSHD', f"{self.BASIN}_OSHD_{self.EXP_ID}_{self.BASIN}_{self.OSHD}_ms{ksoil}.csv")
            #     SWE_source = self.SWE['OSHD'].mean(dim=['lat', 'lon']).loc[slice(f"{self.START_YEAR-1}-10-01", f"{self.END_YEAR}-09-30")]
            #     rfcf = self.pars['Soilpars']['posterior'].loc[ksoil, 'rfcf']
            #     df = process_data(FILE, 1, rfcf, f"{self.START_YEAR-1}-10-01", f"{self.END_YEAR}-09-30", SWE_source)
            #     self.scalars['OSHD'][f"ms{ksoil}"] = df

            for ksoil in range(self.SOIL_K):
                df_list = []
                for year in range(self.START_YEAR, self.END_YEAR + 1):
                    FILE = join(self.OUTDIR, 'OSHD', f"{year}",f"{self.BASIN}_OB_calib_{self.EXP_ID}_{self.BASIN}_{year}_ms{ksoil}.csv")
                    SWE_source = self.SWE['OSHD'].loc[f"{year-1}-10-01":f"{year}-09-30"].mean(dim=['lat', 'lon'])
                    try :
                        rfcf = self.pars['Yearlypars']['rfcf']['OSHD'].loc[ksoil, year]
                    except:
                        rfcf = self.param_fixed['rfcf']                    
                    df = process_data(FILE, 1, rfcf, f"{year-1}-10-01", f"{year}-09-30", SWE_source)
                    df_list.append(df)
                df = pd.concat(df_list)
                self.scalars['OSHD'][f"ms{ksoil}"] = df


            self.scalars['Naive'] = {}
            for ksoil in range(self.SOIL_K):
                FILE = join(self.OUTDIR, 'Naive', f"{self.BASIN}_Naive_{self.EXP_ID}_{self.BASIN}_ms{ksoil}.csv")
                df = process_data(FILE, 1.2, 1.2, f"{self.START_YEAR-1}-10-01", None)
                self.scalars['Naive'][f"ms{ksoil}"] = df
        elif self.SETTING =='Synthetic':
            # FILE = join(self.OUTDIR, 'Synthetic_obs',f"output_Synthetic_obs_{self.BASIN}_{self.EXP_ID}.csv")
            FILE = join(self.OUTDIR, 'Synthetic_obs',f"{self.BASIN}_Synthetic_obs_{self.BASIN}_{self.EXP_ID}.csv")
            df = process_data(FILE, 1, 1, f"{self.START_YEAR-1}-10-01", f"{self.END_YEAR}-09-30")
            self.scalars['Synthetic'] = df


    def compute_yearly_scalars(self):
        #to get the actual rainfall we need to get the original snowfall
        # we do this by dividing the actual snowfall by the sfcf of that year 
        #the original rainfall is then the actual rainfall - the snowfall
        #then we multiply the original rainfall by the correction factor
        if not hasattr(self,'scalars'):
            self.load_scalaroutput()


        for var in ['Q','pet','et','SWE','snowfall','snowmelt','rainfall','Ptotal','Pmean',
                    'satwaterdepth','ustoredepth']:
            self.scalars[var] = {}
            for dist in ['prior','posterior']:
                self.scalars[var][dist] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))
                for year in range(self.START_YEAR,self.END_YEAR+1):
                    for ksoil in range(self.SOIL_K):
                        for kyearly in range(self.YEARLY_K):

                            kk = f"ms{ksoil}_my{kyearly}"
                            self.scalars[var][dist].loc[year,kk] = self.scalars[dist][kk].loc[slice(f"{year-1}-10-01",f"{year}-09-30"),var].sum()
            if self.SETTING == 'Real':
                for dist in ['OSHD','Naive']:
                    self.scalars[var][dist] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))
                    for year in range(self.START_YEAR,self.END_YEAR+1):
                        for ksoil in range(self.SOIL_K):
                            kk = f"ms{ksoil}"
                            self.scalars[var][dist].loc[year,kk] = self.scalars[dist][kk].loc[slice(f"{year-1}-10-01",f"{year}-09-30"),var].sum()
            if self.SETTING == 'Synthetic':
                if var in ['snowfall','snowmelt']:
                    continue
                self.scalars[var]['Synthetic'] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))
                for year in range(self.START_YEAR,self.END_YEAR+1):
                    self.scalars[var]['Synthetic'].loc[year,'Synthetic'] = self.scalars['Synthetic'].loc[f"{year-1}-10-01":f"{year}-09-30",var].sum()




                #OSHD
                # self.scalars[var]['OSHD'] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))

                
                
                
                # self.scalars[var]['Naive'] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))
                # for year in range(self.START_YEAR,self.END_YEAR+1):
                #     for ksoil in range(self.SOIL_K):
                #         kk = f"ms{ksoil}"
                #         self.scalars[var]['Naive'].loc[year,kk] = self.scalars['Naive'][kk].loc[slice(f"{year-1}-10-01",f"{year}-09-30"),var].sum()
        
        
        
        if self.SETTING == 'Synthetic':
            SWE = self.SWE['Synthetic']
            diff = SWE.diff(dim = 'time').fillna(0)

            #set snowfall to zero and out of catchment to nan
            snowmelt = (diff.where(diff<0, 0)*-1)#.mean(dim = ['lat','lon']).to_series()#to_dataframe(name = 'Syn_snowmelt')
            snowmelt = xr.where(np.isnan(self.dem),np.nan,snowmelt)
            snowmelt_ds = snowmelt.mean(dim = ['lat','lon']).to_series()

            #set snowmelt to zero and out of catchment to nan
            snowfall = (diff.where(diff>0, 0))#.mean(dim = ['lat','lon']).to_series()#to_dataframe(name = 'Syn_snowfall')
            snowfall = xr.where(np.isnan(self.dem),np.nan,snowfall)
            snowfall_ds = snowfall.mean(dim = ['lat','lon']).to_series()

            self.scalars['snowmelt']['Synthetic'] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))
            self.scalars['snowfall']['Synthetic'] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))
            # self.scalars['Q']['Synthetic'] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))
            for year in range(self.START_YEAR,self.END_YEAR+1):
                self.scalars['snowmelt']['Synthetic'].loc[year,'Synthetic'] = snowmelt_ds.loc[f"{year-1}-10-01":f"{year}-09-30"].sum()
                self.scalars['snowfall']['Synthetic'] .loc[year,'Synthetic'] = snowfall_ds.loc[f"{year-1}-10-01":f"{year}-09-30"].sum()
                # self.scalars['Q']['Synthetic'] = self.Q_Synthetic.loc[f"{year-1}-10-01":f"{year}-09-30"].sum()
                # self.scalars['snowmelt']['Synthetic'] = snowmelt
                # self.scalars['snowfall']['Synthetic'] = snowfall

                
        self.scalars['Q_obs'] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))
        self.scalars['P_obs'] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))
        # Pobs = xr.open_dataset("/home/pwiersma/scratch/Data/ewatercycle/wflow_Julia_forcing/wflow_MeteoSwiss_1000m_Dischma_2017_2021.nc")['pr']
        # Pobs = xr.where(self.dem.notnull(),Pobs,np.nan)
        # Pobsmean = Pobs.mean(dim = ['lat','lon'])
        # self.scalars['ET_obs'] = pd.DataFrame(index = range(self.START_YEAR,self.END_YEAR+1))
        for year in range(self.START_YEAR,self.END_YEAR+1):
            qobs_y = self.Q_soil['obs'].loc[f"{year-1}-10-01":f"{year}-09-30"].sum() 
            self.scalars['Q_obs'].loc[year,'obs'] =  m3s_to_mm(qobs_y,self.dem_area)
            # self.scalars['P_obs'].loc[year,'obs'] = Pobsmean.loc[f"{year-1}-10-01":f"{year}-09-30"].sum()

    def compute_daily_scalars(self):
        if not hasattr(self,'scalars'):
            self.load_scalaroutput()
        self.daily_scalars = {}
        for var in ['Q','pet','et','SWE','snowfall','snowmelt','rainfall','Ptotal','Pmean',
                    'satwaterdepth','ustoredepth']:
            self.daily_scalars[var] = {}
            for dist in ['prior','posterior']:
                vardic = {}
                for ksoil in range(self.SOIL_K):
                    for kyearly in range(self.YEARLY_K):
                        kk = f"ms{ksoil}_my{kyearly}"
                        vardic[kk] = self.scalars[dist][kk][var]
                self.daily_scalars[var][dist] = pd.DataFrame(vardic)

            if self.SETTING == 'Real':
                for dist in ['OSHD','Naive']:
                    vardic = {}
                    for ksoil in range(self.SOIL_K):
                        kk = f"ms{ksoil}"
                        vardic[kk] = self.scalars[dist][kk][var]
                    self.daily_scalars[var][dist] = pd.DataFrame(vardic)
            if self.SETTING == 'Synthetic':
                if var in ['snowfall','snowmelt']:
                    continue
                self.daily_scalars[var]['Synthetic'] = self.scalars['Synthetic'][var]
        
        if self.SETTING == 'Synthetic':
            SWE = self.SWE['Synthetic']
            diff = SWE.diff(dim = 'time').fillna(0)

            #set snowfall to zero and out of catchment to nan
            snowmelt = (diff.where(diff<0, 0)*-1)#.mean(dim = ['lat','lon']).to_series()#to_dataframe(name = 'Syn_snowmelt')
            snowmelt = xr.where(np.isnan(self.dem),np.nan,snowmelt)
            snowmelt_ds = snowmelt.mean(dim = ['lat','lon']).to_series()

            #set snowmelt to zero and out of catchment to nan
            snowfall = (diff.where(diff>0, 0))#.mean(dim = ['lat','lon']).to_series()#to_dataframe(name = 'Syn_snowfall')
            snowfall = xr.where(np.isnan(self.dem),np.nan,snowfall)
            snowfall_ds = snowfall.mean(dim = ['lat','lon']).to_series()

            # print("snowmelt sum", snowmelt_ds.sum())
            # print("snowfall sum", snowfall_ds.sum())
            # print("romc_all sum",SnowClass(self.BASIN).load_OSHD()['romc_all'].mean(dim = ['x','y']).sum().item())

            self.daily_scalars['snowmelt']['Synthetic'] = snowmelt_ds.fillna(0)
            self.daily_scalars['snowfall']['Synthetic'] = snowfall_ds.fillna(0)
        self.daily_scalars['Q_obs'] = m3s_to_mm(self.Q_soil['obs'],self.dem_area)
        # self.daily_scalars['P_obs'] = self.P_soil['obs']

        
    def plot_waterbalance( self,add_ax = False ):
        if not hasattr(self,'scalars'):
            self.compute_yearly_scalars()
        # I need a seaborn catplot, that makes one figure for each year
        # on the x-axis, i have posterior, Naive and OSHD, one stacked barplot for each
        # to make the stacked barplots, i need to first plot the sum of R and S 
        # then in the same figure, I plot only S on top of it 
        # Then I still need to add the patch for Q+ET 
        # and the uncertainty lines for the ensemble 

        #Lets start with the sum of R and S
        pars_melted = {}
        vars = ['Ptotal','snowfall']
        if self.SETTING == 'Real':
            methods = ['prior','posterior','OSHD','Naive']
        elif self.SETTING == 'Synthetic':
            methods = ['prior','posterior']
        for var in vars:
        # I want to melt the dataframes of prior,posterior,oshd and naive for Ptotal
        #the melted dataframe should have as columns the year, the method and the value
            df_list = []
            for S in methods:
                df = self.scalars[var][S].reset_index(names = 'Year').melt(id_vars = 'Year')
                df['Method'] = self.names[S]
                df_list.append(df)
            df = pd.concat(df_list).reset_index(drop = True)
            pars_melted[var] = df

        # colors = dict(Ptotal = 'tab:blue', snowfall = 'tab:red')
        # saturation = dict(Ptotal = 0.5, snowfall = 1)
        # f1, axes = plt.subplots(1, self.END_YEAR-self.START_YEAR+1, figsize=(15, 8))
        # for year in range(self.START_YEAR,self.END_YEAR+1):
        #     ax = axes[year-self.START_YEAR]
        #     for var in vars:
        #         sns.barplot(data=pars_melted[var][pars_melted[var]['Year'] == year], 
        #         x='Method', y='value', 
        #         hue = 'Method',
        #         # color=colors[var], 
        #         ax=ax,
        #         saturation = saturation[var],
        #         errorbar = ('sd', 0.95))
        #     ax.set_title(year)
        #     ax.legend()

        colors = dict(Ptotal = 'tab:blue', snowfall = 'tab:red')
        saturation = dict(Ptotal = 0.5, snowfall = 1)
        palette = [self.colors['prior'], self.colors['posterior'], self.colors['OSHD'], self.colors['Naive']]

        # WB_out = self.scalars['et']['posterior'].add(self.scalars['Q']['posterior'], fill_value = 0)
        WB_out = self.scalars['et']['posterior'].add(self.scalars['Q_obs']['obs'],axis = 'rows')
        upper_bounds = WB_out.max(axis=1)
        lower_bounds = WB_out.min(axis=1)
        WB_std  = WB_out.std(axis=1)

        if add_ax == False: 
            f1, ax = plt.subplots(1, 1, figsize=(15, 8)) 
        else:
            ax = add_ax

        # from PIL import Image
        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox


        # raindrop = Image.open("/home/pwiersma/scratch/Figures/Raindrop.png")
        # imagebox = OffsetImage(background, zoom=0.5)  # Adjust zoom to scale the image
        # ab = AnnotationBbox(imagebox, (0.5, 0.5), frameon=False)  # (0.5, 0.5) is the data coordinate
        # ax.add_artist(ab)

        for ii,var in enumerate(vars):
            bp = sns.barplot(data=pars_melted[var], 
            x='Year', y='value',  
            hue = 'Method',
            palette = self.colors2,
            # color=colors[var], 
            ax=ax,
            alpha = saturation[var],
            errorbar = ('sd', 0.95),
            dodge = True, gap = 0.1,
             legend = (ii == 1))
        xlim = ax.get_xlim()
        for i,year in enumerate(range(self.START_YEAR,self.END_YEAR+1)):
            xmin = i/len(range(self.START_YEAR,self.END_YEAR+1))
            xmax = (i+1)/len(range(self.START_YEAR,self.END_YEAR+1))
            med = WB_out.loc[year].median()
            std = self.scalars['et']['posterior'].loc[year].median() * 0.2
            # std = WB_out.loc[year].std()
            Q_std = self.scalars["Q_obs"].loc[year,'obs']*0.05
            #plot Q_observed 
            ax.hlines(med, color = 'black', linestyle = 'dashed',
                       xmin =i-0.5,xmax =i+0.5, label = r'$Q_{obs} + ET$'* (i == 0))
            
            if self.SETTING =='Synthetic':
                ax.hlines(self.scalars['snowfall']['Synthetic'].loc[year],
                color = 'black', linestyle = (0, (1, 5)),
                xmin =i-0.5,xmax =i+0.5, label = 'Reference snowfall'* (i == 0),
                linewidth = 2)


            # ax.hlines(self.scalars['P_obs'].loc[year,'obs'], color = 'black', linestyle = 'dotted',
            #            xmin =i-0.5,xmax =i+0.5, label = 'P_obs'* (i == 0))
                      # ax.fill_between([i-0.5,i+0.5], lower_bounds.loc[year], upper_bounds.loc[year], alpha=0.3, color='gray')
            color = 'None'
            p1 = ax.fill_between([i-0.5,i+0.5], med-std, med+std, alpha=0.3, facecolor=color,
                            hatch = '///', label = 'ET-Uncertainty'* (i == 0))
            p2 = ax.fill_between([i-0.5,i+0.5], med-std, med-std-Q_std, alpha=0.2, facecolor=color,
                            hatch = "\\\\\\", label = 'Q-Uncertainty'* (i == 0))
            p3 = ax.fill_between([i-0.5,i+0.5], med+std, med+std+Q_std, alpha=0.2, facecolor=color,
                            hatch = "\\\\\\")
        ax.get_legend_handles_labels()
        ax.legend(loc = 'lower right', ncol = 2)
        ax.set_ylabel('Water balance component \n [mm / year]')      



        #add the text "snowfall" and "rainfall" to the first bars
        ax.text(0, 0.66 * med, 'Rainfall', horizontalalignment='center', verticalalignment='center',
        bbox=dict(facecolor='white', alpha=0.5))#, transform=ax.transAxes)
        ax.text(0, 0.2*med, 'Snowfall', horizontalalignment='center', verticalalignment='center',
        bbox=dict(facecolor='white', alpha=0.5))#, transform=ax.transAxes)

        ax.set_xlim(ax.get_xlim()[0]+0.25,ax.get_xlim()[1]-0.25)

        if ax ==False:
            ax.set_title(f"{self.BASIN} - Water balance")
        ax.set_xlabel(None)

        if ax == False:
            plt.savefig(join(self.FIGDIR,f"{self.BASIN}_Waterbalance.png"), bbox_inches='tight', dpi=200)
        else:
            return ax


        #now make a catplot 
        # f1,ax1 = plt.subplots(1,1,figsize = (10,5))
        # p1 = sns.catplot(data = pars_melted['Ptotal'], x = 'Method', col = 'Year', y = 'value', color = 'tab:blue', kind = 'bar', ax = ax1)
        # p2 = sns.catplot(data = pars_melted['snowfall'], x = 'Method', col = 'Year', y = 'value', color = 'tab:red', kind = 'bar', ax = ax1)

        # a1 = p1.facet_axis(0,0)
        # a2 = p2.facet_axis(0,0)

        # f1,ax1 = plt.subplots(1,1,figsize = (10,5))
        # #add a1 and a2 to ax1
        # f1.axes.append(a1)

        # f1, ax1 = plt.subplots(1, 1, figsize=(10, 5))

        # # Extract the axes from p1 and p2
        # axes_p1 = p1.axes.flatten()
        # axes_p2 = p2.axes.flatten()

        # # Add the axes from p1 and p2 to the new figure
        # for ax in axes_p1:
        #     for child in ax.get_children():
        #         if isinstance(child, plt.Axes):
        #             f1._axstack.add(f1._make_key(ax), child)

        # for ax in axes_p2:
        #     for child in ax.get_children():
        #         if isinstance(child, plt.Axes):
        #             f1._axstack.add(f1._make_key(ax), child)




    def plot_groundwater(self):
        if not hasattr(self,'scalars'):
            self.compute_daily_scalars()
        # I want to plot the sum of the saturated and unsaturated storage

        unsatstore = self.daily_scalars['ustoredepth']
        satstore = self.daily_scalars['satwaterdepth']

        blues = sns.color_palette('Blues_r', 2)
        reds = sns.color_palette('Greys_r', 2)

        f1, ax1 = plt.subplots(1, 1, figsize=(10, 5))

        # Plot the posterior data with a single legend entry
        posterior_unsat = ax1.plot(unsatstore['posterior'], color=blues[0], label='Q-SWE S_unsat')
        posterior_sat = ax1.plot(satstore['posterior'], color=blues[1],label = 'Q-SWE S_sat')

        if self.SETTING == 'Synthetic':
            # Plot the synthetic (observed) data with individual legend entries
            synthetic_unsat, = ax1.plot(unsatstore['Synthetic'], color=reds[0], label='Synthetic S_unsat')
            synthetic_sat, = ax1.plot(satstore['Synthetic'], color=reds[1],label='Synthetic S_sat')

        # Combine the legends
        handles, labels = ax1.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax1.legend(unique_labels.values(), unique_labels.keys())

        # Add titles and labels
        ax1.set_title('Storage Comparison')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Storage')

        # Add grid for better readability
        ax1.grid(True)

        # Show the plot
        plt.show()

        f1.savefig(join(self.FIGDIR,f"{self.BASIN}_Groundwater.png"), bbox_inches='tight', dpi=200)

    def load_pixel_scalars(self,ilat,ilon):
        def rename_column(col):
            parts = col.split('_')
            return '_'.join(parts[-3:])
        

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
        def fix_dfs(df):
            """
            Fix the dataframes by removing NaN values and duplicate indices"""
            index = df.index.unique()
            newdf = pd.DataFrame(index = index)
            for col in df.columns:
                data = df[col].values
                data = data[~np.isnan(data)]
                newdf[col] = data
            return newdf
        # vars = ['snow','snowwater','actevap','leakage','rfmelt','canopystorage','interception'
        #         ,'satstore','unsatstore','canopy_evap_fraction','throughfall','runoff',
        #         'Q_river','Qin_river','inwater_river','Q_land','Qin_land','inwater_land',
        #         'to_river_land','ssfself.N','ssfin','to_river_ssf']
        # vars = ['snow','snowwater','actevap','leakage','rfmelt','canopystorage','interception'
        # ,'satstore','unsatstore','canopy_evap_fraction','throughfall','runoff',
        # 'Q_river','Qin_river','inwater_river','Q_land','Qin_land','inwater_land',
        # 'to_river_land','ssf','ssfin','to_river_ssf',
        #                 'transfer','vwc','zi','unsatstore_layer','act_thickl']
        vars = self.NC_STATES

        # Initialize dictionaries to store dataframes for each variable
        self.pixel_scalars = {}
        prior_dfs = {}
        posterior_dfs = {}

        # Define the ranges for years, ksoil, and variables
        years = range(self.START_YEAR, self.END_YEAR+1)  # Example range of years
        ksoils = range(0, self.SOIL_K)       # Example range of ksoil values
        # vars = ['rfmelt', 'snow']  # List of variables

        coords = self.dem.isel(lat=ilat, lon=ilon)
        lat,lon = coords.lat.item(),coords.lon.item()

        #convert vwc to vwc_1,vwc_2,vwc_3, vwc_4 within the vars list 
        for var in ['vwc','ustorelayerdepth','zi','act_thickl']:
            if var in vars:
                vars.remove(var)
                vars.extend([f"{var}_{i}" for i in range(1,5)])

        #Yearlycalib postrun results
        for var in vars:
            prior_list = []
            posterior_list = []
            
            for year in years:
                for ksoil in ksoils:
                    folder = join(self.OUTDIR, "Yearlycalib", f"{year}", f"{ksoil}")
                    basevar = var[:-2] if (var[-2] == '_' and var[-1] != 'r') else var
                    file = join(folder, f"{self.BASIN}_{basevar}.nc")
                    
                    if os.path.isfile(file):
                        # print("Loading ",var)
                        data = xr.open_dataset(file).sel(lat=lat, lon=lon)
                        if 'layer' in data.dims:
                            data = data.sel(layer = var[-1]).drop_vars('layer')
                        #     datadic = {int(layer.item()): data.sel(layer=layer.item()) for layer in data.layer}
                        # else:
                        #     datadic= {0:data}
                        # for key,value in datadic.items():
                        #     data = value
                        df = data.to_pandas()
                        df = df.drop(['lat', 'lon'], axis=1)

                        if var in ['to_river_ssf','ssf','ssfin']:
                            df = m3d_to_mm(df, self.grid_area)
                        elif var in ['Q','qin_land','qin_river','q_river','inwater_river','inwater_land','to_river_land']:
                            df = m3s_to_mm(df, self.grid_area)

                        prior = df.filter(like='prior')
                        prior.columns = [rename_column(col) for col in prior.columns]
                        prior_list.append(prior)
                        
                        posterior = df.filter(like='posterior')
                        posterior.columns = [rename_column(col) for col in posterior.columns]
                        posterior_list.append(posterior)
                    else:
                        print(f"File {file} not found")
                        break
                
            # Concatenate all dataframes for the current variable
            prior_dfs[var] = fix_dfs(pd.concat(prior_list, ignore_index=False))
            posterior_dfs[var] = fix_dfs(pd.concat(posterior_list, ignore_index=False))
        prior_dfs['SWE'] = prior_dfs['snow'] + prior_dfs['snowwater']
        posterior_dfs['SWE'] = posterior_dfs['snow'] + posterior_dfs['snowwater']
        self.pixel_scalars['prior'] = prior_dfs
        self.pixel_scalars['posterior'] = posterior_dfs

        #Soilcalib results 
        self.pixel_scalars['Soilcalib'] = {}
        prior_dfs = {}
        posterior_dfs = {}
        for var in vars:
            folder = join(self.OUTDIR, "Soilcalib")
            basevar = var[:-2] if (var[-2] == '_' and var[-1] != 'r') else var
            file = join(folder, f"{self.BASIN}_{basevar}.nc")
            data = xr.open_dataset(file).sel(lat=lat, lon=lon)
            if 'layer' in data.dims:
                data = data.sel(layer = var[-1]).drop_vars('layer')         
            df = data.to_pandas()   
            df = df.drop(['lat', 'lon'], axis=1)
            if var in ['to_river_ssf','ssf','ssfin']:
                df = m3d_to_mm(df, self.grid_area)
            elif var in ['Q','qin_land','qin_river','q_river','inwater_river','inwater_land','to_river_land']:
                df = m3s_to_mm(df, self.grid_area)
            prior = df.filter(like='prior')
            prior.columns = [rename_column(col) for col in prior.columns]
            prior_dfs[var] = prior
            posterior = df.filter(like='posterior')
            posterior.columns = [rename_column(col) for col in posterior.columns]
            posterior_dfs[var] = posterior
        prior_dfs['SWE'] = prior_dfs['snow'] + prior_dfs['snowwater']
        posterior_dfs['SWE'] = posterior_dfs['snow'] + posterior_dfs['snowwater']
        self.pixel_scalars['Soilcalib']['prior'] = prior_dfs
        self.pixel_scalars['Soilcalib']['posterior'] = posterior_dfs

        #for SYN
        if self.SETTING == 'Synthetic':
            syn_dfs = {}
            for var in vars:
                folder = join(self.OUTDIR, "Synthetic_obs")
                basevar = var[:-2] if (var[-2] == '_' and var[-1] != 'r') else var
                file = join(folder, f"{self.BASIN}_{basevar}.nc")
                data = xr.open_dataset(file).sel(lat=lat, lon=lon)
                if 'layer' in data.dims:
                    data = data.sel(layer = var[-1]).drop_vars('layer')
                df = data.to_pandas()                  
                df = df.drop(['lat', 'lon'], axis=1)
                df.columns = ['Synthetic']
                if var in ['to_river_ssf','ssf','ssfin']:
                    df = m3d_to_mm(df, self.grid_area)
                elif var in ['Q','qin_land','qin_river','q_river','inwater_river','inwater_land','to_river_land']:
                    df = m3s_to_mm(df, self.grid_area)
                        
                syn_dfs[var] = df
            if self.SYN_SNOWMODEL =='OSHD':
                syn_dfs['SWE'] = self.SWE['Synthetic'].sel(lat=lat, lon=lon).to_pandas()
            elif self.SYN_SNOWMODEL in ['seasonal','Hock']:
                syn_dfs['SWE'] = syn_dfs['snow'] + syn_dfs['snowwater']
            self.pixel_scalars['Synthetic'] = syn_dfs

    def plot_pixel_scalars(self,ilat,ilon,timerange):
        # if not hasattr(self,'pixel_scalars'):

        # ilat = 5
        # ilon = 5
        self.load_pixel_scalars(ilat,ilon)

        posterior = self.pixel_scalars['posterior']
        posterior_sc = self.pixel_scalars['Soilcalib']['posterior']
        synthetic = self.pixel_scalars['Synthetic']
       
        f1, axes = plt.subplots(4, 2, figsize=(12, 14),sharey = 'row')
        axes = axes.flatten()

        # Plot the DEM
        coords = self.dem.isel(lat=ilat, lon=ilon)

        self.dem.plot(ax=axes[0], add_colorbar=False)
        axes[0].plot(coords.lon, coords.lat, 'rx', label='Selected cell')
        axes[0].set_title(f"Elevation {int(coords.values.item())} m")
        axes[0].legend()

        axes[1].axis('off')
        #add text to axes[1] which prints the soil parameters
        soil_paramtext = ""
        for key in self.trueparams:
            try: 
                if self.param_kinds[key] == 'soil':
                    soil_paramtext+= (f"{key} = {self.trueparams[key]} \n ")
            except:
                continue
        text = f"True soil parameters:\n {soil_paramtext}"
        axes[1].text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
        # axes[1].T  # Turn off the second DEM plot

        # Define the variables to plot
        variables = {
            'Vertical fluxes': ['rainfallplusmelt', 'transfer', 'actevap', 'runoff'],
            'Storage': ['SWE', 'satwaterdepth', 'ustorelayerdepth_1', 'ustorelayerdepth_2', 'ustorelayerdepth_3', 'ustorelayerdepth_4'],
            'Lateral fluxes': ['ssf', 'ssfin', 'to_river_ssf', 'netQriver', 'netQland', 'to_river_land', 'inwater_land', 'inwater_river']
        }

        # Plot the variables
        for i, (title, vars) in enumerate(variables.items(), start=1):
            for j, dataset in enumerate(['Synthetic', 'Posterior Median']):
                for var in vars:
                    if var in ['netQriver', 'netQland']:
                        if dataset == 'Synthetic':
                            netQriver = synthetic['q_river'].loc[timerange]['Synthetic'] - synthetic['qin_river'].loc[timerange]['Synthetic']
                            netQland = synthetic['q_land'].loc[timerange]['Synthetic'] - synthetic['qin_land'].loc[timerange]['Synthetic']
                        else:
                            netQriver = posterior['q_river'].loc[timerange].median(axis = 1) - posterior['qin_river'].loc[timerange].median(axis = 1)
                            netQland = posterior['q_land'].loc[timerange].median(axis = 1) - posterior['qin_land'].loc[timerange].median(axis = 1)
                        netQriver.plot(ax=axes[i*2+j], label='netQriver')
                        netQland.plot(ax=axes[i*2+j], label='netQland')
                    else:
                        if dataset == 'Synthetic':
                            if not var=='SWE':
                                synthetic[var].loc[timerange]['Synthetic'].plot(ax=axes[i*2+j], label=var)
                            else:
                                synthetic[var].loc[timerange].plot(ax=axes[i*2+j], label=var)
                        else:
                            posterior[var].loc[timerange].median(axis = 1).plot(ax=axes[i*2+j], label=var)
                axes[i*2+j].legend()
                axes[i*2+j].set_title(f'{title} ({dataset})')
                axes[i*2+j].set_ylabel('mm/day' if 'fluxes' in title else 'mm')
                axes[i*2+j].grid()

        plt.tight_layout()
        # plt.show()
        f1.savefig(join(self.FIGDIR,f"{self.BASIN}_Pixel_scalars_{ilat}_{ilon}.png"), bbox_inches='tight', dpi=200)



    
    def load_MODIS(self):
        if self.RESOLUTION == '1000m':
            res = '1km'
        else:
            res = self.RESOLUTION
        MOD10A_file = f"/home/pwiersma/scratch/Data/SnowCover/MOD10A_2000_2022_{res}_{self.BASIN}.nc"
        MOD10A = xr.open_dataset(MOD10A_file).rename(dict(NDSI_Snow_Cover='NDSI', NDSI_Snow_Cover_Class='Class'))
        mask = MOD10A['Class'].isin([250, 200, 201, 211, 254, 255, 237])
        NDSI = xr.where(~mask, MOD10A['NDSI'], np.nan)
        obsbinary = xr.where(NDSI > 30, 1, NDSI * 0)
        obsbinary = obsbinary.sel(time=slice(f'{self.START_YEAR-1}-10-01', f'{self.END_YEAR}-09-30'))
        self.obsbinary = obsbinary
        return obsbinary

    def convert_SWE_to_snow_cover(self,SWE_threshold=10):
        binary_snow_cover_dict = {}
        for key in self.SWE:
            binary_snow_cover = xr.where(self.SWE[key] > SWE_threshold, 1, 0)
            binary_snow_cover_dict[key] = binary_snow_cover
        return binary_snow_cover_dict

    def evaluate_snow_cover(self, obsbinary, SWE_threshold=10):
        masked_binary = xr.where(np.isnan(self.dem), np.nan, obsbinary)
        observed_pixels = xr.where(~np.isnan(masked_binary), 1, np.nan)
        obspix_spatial = observed_pixels.sum(dim='time')
        obspix_temporal = observed_pixels.sum(dim=['lon', 'lat'])
        obspix_monthly = observed_pixels.groupby('time.month').sum()
        total_pixels = obspix_spatial.sum().item()

        evaldic = {}
        evaldic['SC'] = {}
        evaldic['SC']['obs_masked_binary'] = masked_binary

        self.SC = {}
        self.SC['obs'] = masked_binary 
        binary_snow_cover_dict = self.convert_SWE_to_snow_cover(SWE_threshold)
        for key in binary_snow_cover_dict:
            print(key)
            self.SC[key] = {}
            self.SC[key]['binary'] = binary_snow_cover_dict[key]

            d = self.SC[key]
            d['sc_diff'] = d['binary'] - masked_binary
            d['spatial_error'] = np.abs(d['sc_diff']).sum(dim='time')
            d['total_error'] = d['spatial_error'].sum().to_pandas()
            d['total_OA'] = (1 - d['total_error'] / total_pixels) * 100
            d['OA_daily'] = (1 - (np.abs(d['sc_diff']).sum(dim=['lat', 'lon']).groupby(
                'time.month').sum() / obspix_temporal.groupby('time.month').sum())).drop_vars('spatial_ref').to_pandas()
            d['OA_monthly'] = (1 - (np.abs(d['sc_diff']).groupby(
                'time.month').sum() / obspix_monthly)).drop_vars('spatial_ref')
            d['FP_monthly'] = ((d['sc_diff'] == 1).groupby('time.month').sum() / obspix_monthly).drop_vars('spatial_ref')
            d['FN_monthly'] = ((d['sc_diff'] == -1).groupby('time.month').sum() / obspix_monthly).drop_vars('spatial_ref')
        return self.SC 
    
    def calculate_overall_accuracy(self):
        OA_list = []
        for year in range(self.START_YEAR , self.END_YEAR + 1):
            spatial_dic = self.SC
            dic = {}
            for key, d in spatial_dic.items():
                if key in ['prior', 'posterior']:#, 'Naive', 'OSHD']:
                    print(key)
                    # data = d['total_OA'] 
                    # for col in data.columns:
                    #     dic[key + '_' + col] = data[col]
                    dic[key] = d['total_OA']
                elif key in ['Naive', 'OSHD']:
                    dic[key] = pd.Series(data = d['total_OA'], index = [key], name = key)
            OA_all = pd.DataFrame(dic)
            OA_all['combined'] = OA_all['Naive'].combine_first(OA_all['posterior']).combine_first(OA_all['prior']).combine_first(OA_all['OSHD'])
            OA_all = OA_all.drop(['prior', 'posterior', 'Naive', 'OSHD'], axis=1)
            
            OA_all['Model'] = OA_all['combined'] * np.nan
            OA_all['Year'] = str(year)
            for ID in OA_all.index:
                if ('posterior' in ID):
                    OA_all.loc[ID, 'Model'] = 'posterior'
                elif ('prior' in ID):
                    OA_all.loc[ID, 'Model'] = 'prior'
                elif ID == 'OSHD':
                    OA_all.loc[ID, 'Model'] = 'OSHD'
                elif ID == 'Naive':
                    OA_all.loc[ID, 'Model'] = 'Naive'
            melted_OA = pd.melt(OA_all, id_vars=['Model', 'Year'])
            OA_list.append(melted_OA)
        OA_concat = pd.concat(OA_list)
        self.OA = OA_concat
        return OA_concat

    def load_SC(self):
        self.load_MODIS()
        self.SC = self.evaluate_snow_cover(self.obsbinary)
        self.OA = self.calculate_overall_accuracy()

    def plot_OA(self):
        if 'SC' not in self.__dict__:
            self.load_SC()
        f1,ax = plt.subplots(1,1,figsize = (12,6))#,sharex = True)
        # plt.subplots_adjust(hspace = 0.5)
        # sns.set(style = 'ticks')
        OA_concat = self.OA
        sns.stripplot(ax =ax, data =OA_concat,
                    x = 'value', hue = 'Model', alpha = 0.2,
                    dodge =True, legend = False, marker = 'd',size = 6,
                    palette = "colorblind", jitter = 0.2, linewidth = 0.2
                    )
        sns.boxplot(ax =ax, data = OA_concat,
                    x = 'value', hue = 'Model', fill = False,
                    dodge = True, legend = "brief",saturation = 0.75,
                    palette = "colorblind",
                    gap = 0.1
                    )#y = 'Year',
        # ax = sns.boxplot(data = OA_final,x = 'value',y = 'Basin',hue = 'Model',orient = 'h',ax = ax)
        ax.set_xlabel('Mean overall Accuracy [%]')
        ax.set_xlim(right = 100)
        # ax.set_ylabel('Model setting')
        ax.set_title(f"{self.BASIN} - Yearly MODIS evaluation ({self.START_YEAR} - {self.END_YEAR})",fontdict = {'fontsize':'xx-large'})
        # ax.set_xlim(0.5,0.8)
        ax.grid(axis = 'x')
        ax.legend(title = None)
        plt.savefig(join(self.FIGDIR,f"{self.BASIN}_Yearly_OA.png"), bbox_inches='tight', dpi=200)


    def plot_snowmelt(self):
        snowmelt = self.daily_scalars['snowmelt']
        # yearlyQdir = join(self.FIGDIR,'YearlySnowmelt')
        # if not os.path.exists(yearlyQdir):
        #     os.makedirs(yearlyQdir)


        f1, axes = plt.subplots(self.END_YEAR-self.START_YEAR+1,1,  figsize=(15, 20))
        for year in range(self.START_YEAR,self.END_YEAR+1):
            ax = axes[year-self.START_YEAR]
            timeslice = slice(f"{year}-03-01",f"{year}-07-30")
            
            if self.SETTING == 'Synthetic':
                dic = dict(Q = [snowmelt['Synthetic'].loc[timeslice],
                                snowmelt['prior'].loc[timeslice],
                                snowmelt['posterior'].loc[timeslice]],
                                # self.Q_OSHD.loc[timeslice],
                                # self.Q_Naive.loc[timeslice]],
                    colors = ['black',self.colors['prior'],self.colors['posterior']],
                        labels = ['Synthetic obs (OSHD)','Prior','Posterior'],
                        alphas = [1,self.alphas['prior'],self.alphas['posterior']],
                        kind = ['obs','lines','lines'],
                        linestyle = ['','--','-'],
                        zorder = [4,3,5],
                        sizes = [1,1,1],
                        timeslice = timeslice)
            ax = self.hydrographs_fromdic(hydrograph_dic = dic,
                                         grid = True, 
                                         title = f"{self.BASIN} - Snow melt {year}",
                                         ax = ax,
            )
            ax.set_xlabel(None)
            ax.legend(loc = 'upper right')
            ax.set_ylim(bottom = 0, top = 40)
            if year != self.END_YEAR: 
                # ax.set_xticks([])
                ax.set_xticklabels([])
                ax.legend().remove()
        plt.savefig(join(self.FIGDIR,f"{self.BASIN}_snowmelt_allyears.png"), bbox_inches='tight', dpi=200)




        
    def hydrographs_soilcalib(self):
        """
        Plot prior and posterior hydrographs for soil calibration        
        """
        soilQdir = join(self.FIGDIR,'SoilQ')
        if not os.path.exists(soilQdir):
            os.makedirs(soilQdir)

        for year in range(self.START_YEAR,self.END_YEAR+1):
            timeslice = slice(f"{year-1}-10-01",f"{year}-09-30")
            
            if self.SETTING == 'Real':
                dic = dict(Q = [self.Q_soil['obs'].loc[timeslice],
                                self.Q_soil['prior'].loc[timeslice],
                                self.Q_soil['posterior'].loc[timeslice]],
                    colors = ['black','tab:orange','tab:blue'],
                        labels = ['Obs','Prior','Posterior'],
                        alphas = [1,0.5,0.5],
                        kind = ['obs','lines','lines'],
                        linestyle = ['','--','-'],
                        zorder = [0,1,2],
                        sizes = [1,1,1])
            elif self.SETTING == 'Synthetic':
                dic = dict(Q = [self.Q_Synthetic.loc[timeslice],
                                self.Q_soil['prior'].loc[timeslice],
                                self.Q_soil['posterior'].loc[timeslice]],
                    colors = ['black','tab:orange','tab:blue'],
                        labels = ['Obs','Prior','Posterior'],
                        alphas = [1,0.5,0.5],
                        kind = ['obs','lines','lines'],
                        linestyle = ['','--','-'],
                        zorder = [0,1,2],
                        sizes = [1,1,1])


            f1,ax1 = plt.subplots(figsize = (12,4))
            ax1 = self.hydrographs_fromdic(hydrograph_dic = dic,
                                         grid = True, 
                                         title = f"{self.BASIN} - Soil calibration {year}",
                                         ax = ax1,
            )
            plt.savefig(join(soilQdir,f"{self.BASIN}_Q_{year}.png"), bbox_inches='tight', dpi=200)

    
    def hydrographs_yearlycalib(self):
        """
        Plot prior and posterior hydrographs for yearly calibration        
        """
        yearlyQdir = join(self.FIGDIR,'YearlyQ')
        if not os.path.exists(yearlyQdir):
            os.makedirs(yearlyQdir)


        # f1, axes = plt.subplots(self.END_YEAR-self.START_YEAR+1,1,  figsize=(15, 20))
        for year in range(self.START_YEAR,self.END_YEAR+1):
            # ax = axes[year-self.START_YEAR]
            timeslice = slice(f"{year}-03-01",f"{year}-09-30")
            
            if self.SETTING == 'Real':
                dic = dict(Q = [self.Q_yearly['obs'].loc[timeslice],
                                self.Q_yearly['prior'].loc[timeslice],
                                self.Q_yearly['posterior'].loc[timeslice],
                                self.Q_OSHD.loc[timeslice],
                                self.Q_Naive.loc[timeslice]],
                    colors = ['black',self.colors['prior'],self.colors['posterior'],self.colors['OSHD'],self.colors['Naive']],
                        labels = ['Obs','Prior','Posterior','Q-OSHD','Q-Naive'],
                        alphas = [1,self.alphas['prior'],self.alphas['posterior'],
                                  self.alphas['OSHD'],self.alphas['Naive']],
                        kind = ['obs','lines','lines','lines','lines'],
                        linestyle = ['','--','-','-','-'],
                        zorder = [4,3,5,2,0],
                        sizes = [1,1,1,1,1],
                        timeslice = timeslice)
            
            elif self.SETTING == 'Synthetic':
                dic = dict(Q = [self.Q_Synthetic.loc[timeslice],
                                self.Q_yearly['prior'].loc[timeslice],
                                self.Q_yearly['posterior'].loc[timeslice]],
                    colors = ['black','tab:orange','tab:blue'],
                        labels = ['Obs','Prior','Posterior'],
                        alphas = [1,0.5,0.5],
                        kind = ['obs','lines','lines'],
                        linestyle = ['','--','-'],
                        zorder = [0,1,2],
                        sizes = [1,1,1],
                        timeslice = timeslice)


            # snowmelt = dict()
            f1,ax = plt.subplots(figsize = (12,6))            
            ax =self.hydrographs_fromdic(hydrograph_dic = dic,
                                           filtered = self.Q_filtered[str(year)]['obs'][timeslice],
                                         figsize = (12,6),
                                         grid = True, 
                                         title = f"{self.BASIN} - Yearly calibration {year}",
                                         ax = ax,
                                         snowmonths_hatch = True,
                                            snowmelt = self.daily_scalars['snowmelt']
            )
            plt.savefig(join(yearlyQdir,f"{self.BASIN}_Q_{year}.png"), bbox_inches='tight', dpi=200)
        
        # plt.savefig(join(yearlyQdir,f"{self.BASIN}_Q_{year}_allyears.png"), bbox_inches='tight', dpi=300)

    
   
    def hydrographs_fromdic(self,
                            hydrograph_dic: dict,
        *,
        perturbed:str = None,
        filtered: pd.DataFrame = None,
        # reference: str,
        precipitation: pd.DataFrame = None,
        sim_names = list,
        dpi: int = None,
        title: str = "Hydrograph",
        discharge_units: str = "m$^3$ s$^{-1}$",
        precipitation_units: str = "mm day$^{-1}$",
        figsize: Tuple[float, float] = (10, 10),
        filename: Union[os.PathLike, str] = None,
        grid : bool = True,
        legend_loc : Tuple[float,float] = (1.10,1),
        ax : plt.Axes = None,
        snowmonths_hatch: bool = False,
        snowmelt: pd.DataFrame = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:

        # fig, ax = plt.subplots(
        #     dpi=dpi,
        #     figsize=figsize
        # )

        ax.set_title(title)
        ax.set_ylabel(f"Discharge ({discharge_units})")

        d = hydrograph_dic 
        labels = d.get('labels')
        colors = d.get('colors')
        kind = d.get('kind') #line or spread
        alphas = d.get('alphas')
        sizes  = d.get('sizes')
        zorder = d.get('zorder')
        linestyles = d.get('linestyle')
        timeslice = d.get('timeslice')
        Q = d.get('Q')
            
        for i,q in enumerate(Q):    
            if kind[i] =='obs':
                if filtered is not None:
                    q.plot(ax=ax,
                       color = 'black',
                       linestyle = 'dotted',
                       label = labels[i],
                       alpha = 0.4,
                       zorder = 100)#, **kwargs) 
                    filtered.plot(ax=ax,
                                  color = 'black',
                                  zorder = 99,
                                    label = f"Filtered {labels[i]}",
                                  linestyle = 'dashed')
                else:
                    q.plot(ax=ax,
                        color = 'black',
                        linestyle = 'dashed',
                        label = labels[i],
                        alpha = alphas[i],
                        zorder = 100)#, **kwargs) 
                    
            else:
                cols = q.columns 

                y_sim_min = q[cols].min(axis = 1)
                y_sim_max = q[cols].max(axis = 1)

                y_sim_median = q[cols].median(axis = 1)

                if kind[i] == 'spread':
                    ax.fill_between(y_sim_min.index,
                                    y_sim_min,
                                    y_sim_max,
                                    alpha = alphas[i],
                                    color = colors[i],
                                    label = labels[i],
                                    linewidth = sizes[i],               
                                    zorder = zorder[i])
                    if i ==1:
                        medianplot = ax.plot(y_sim_median.index,y_sim_median,
                                            color = colors[i],
                                            alpha = min(alphas[i] + 0.1,1))
                elif kind[i] == 'line':
                    medianplot = ax.plot(y_sim_median.index,y_sim_median,
                                    color = colors[i],
                                    alpha = min(alphas[i] + 0.1,1),
                                    label = labels[i],
                                    linestyle = linestyles[i],
                                    linewidth = sizes[i],
                                    zorder = zorder[i])
                elif kind[i] == 'lines':   
                    for n,col in enumerate(cols):
                        if n ==0:
                            label = labels[i]
                        else:
                            label = None
                        ax.plot(q[col].index,q[col],
                                    color = colors[i],
                                    alpha = alphas[i],
                                    label = label,
                                    linewidth = sizes[i],
                                    zorder = zorder[i])
                        
        handles, labels = ax.get_legend_handles_labels()

        # Add precipitation as bar plot to the top if specified
        if precipitation is not None:
            ax_pr = ax.twinx()
            ax_pr.invert_yaxis()
            ax_pr.set_ylabel(f"Precipitation ({precipitation_units})")
            # prop_cycler = ax._get_lines.prop_cycler

            # for pr_label, pr_timeseries in precipitation.iteritems():
            #     #color = next(prop_cycler)["color"]
            color = 'tab:blue'
            pr_timeseries =precipitation
            ax_pr.bar(
                pr_timeseries.index.values,
                pr_timeseries.values[:,0],
                alpha=0.4,
                label='MeteoSwiss P',
                color=color,
            )

            # tweak ylim to make space at bottom and top
            ax_pr.set_ylim(ax_pr.get_ylim()[0] * (7 / 2), 0)
            # ax.set_ylim(0, ax.get_ylim()[1] * (7 / 5))

            # prepend handles/labels so they appear at the top
            handles_pr, labels_pr = ax_pr.get_legend_handles_labels()
            handles = handles_pr + handles
            labels = labels_pr + labels
        
        if snowmelt is not None:
            ax_sm = ax.twinx()
            ax_sm.invert_yaxis()
            # ax_sm.set_ylabel(f"Snowmelt (mm/day)")
            ax_sm.set_ylabel(f"Snowmelt (m$^3$/s)")
            
            for S in snowmelt.keys():
                SM = snowmelt[S][timeslice]
                #convert to m3/s
                SM = SM * 1e-3 * self.dem_area * 1e6 * (1/(24*60*60))     #SM * 60*60*24 *(1/self.dem_area) 
                color = self.colors[S]
                if S in ['posterior','prior','OSHD','Naive']:
                    linestyle = '-'
                    labelflag = False
                    for col in SM.columns:
                        if labelflag == False:
                            label = S
                            labelflag = True
                        else:
                            label = "_noLegend"
                        ax_sm.plot(SM.index,SM[col].values, color=color, 
                                   label="_noLegend", linestyle=linestyle,
                                   alpha = self.alphas[S])
                if S=='Synthetic':
                    linestyle = '--'
                    ax_sm.plot(SM.index,SM.values, color=color, label='_noLegend',
                               linestyle=linestyle)



            # tweak ylim to make space at bottom and top
            ax_sm.set_ylim(ax_sm.get_ylim()[0] * (7 / 2), 0)
            # ax.set_ylim(0, ax.get_ylim()[1] * (7 / 5))
            ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
            # ax_sm.set_ylim(ax.get_ylim()[1] * 1.1,0)

            # prepend handles/labels so they appear at the top
            handles_sm, labels_sm = ax_sm.get_legend_handles_labels()
            handles = handles_sm + handles
            labels = labels_sm + labels

            #add a text box with the total snowmelt
            print("Snowmelt sum", SM.sum().item())
            print("Q sum ", Q[0].sum().item())

        # Put the legend outside the plot
        ax.legend(handles, labels, bbox_to_anchor=legend_loc, loc="upper left")

        if snowmonths_hatch:
            year = q.index[-1].year
            first_date = q.index[0]
            hatch_start = (datetime(year, self.DOUBLE_MONTHS[0], 1)-first_date).days+ax.get_xlim()[0]
            hatch_end = (datetime(year, self.DOUBLE_MONTHS[-1]+1,1)-first_date).days+ax.get_xlim()[0]
            # print(hatch_start,hatch_end)
            pattern_patch = patches.Rectangle(
                        (hatch_start, 0), hatch_end-hatch_start, ax.get_ylim()[1],
                          transform=ax.transData, 
                        facecolor='none', edgecolor='black', alpha = 0.05,
                        hatch='*')  # Diagonal lines pattern)          
            ax.add_patch(pattern_patch)

        # set formatting for xticks
        date_fmt = DateFormatter("%Y-%m")
        ax.xaxis.set_major_formatter(date_fmt)
        ax.tick_params(axis="x", rotation=30)
        if grid ==True:
            ax.grid()
            
        # if filename is not None:
        #     fig.savefig(filename, bbox_inches="tight", dpi=dpi)
        if snowmelt is not None:
            return ax,ax_sm
        elif precipitation is not None:
            return ax,ax_pr
        else:
            return ax

    def plot_SWE_as_ax(ax):
        S = self.SWE['OSHD'].isel(time = 90)
        S = xr.where(np.isnan(self.dem),np.nan,S)
        S.plot(ax = ax,alpha = 0.7)


    def plot_dem_with_hillshade(ax,dem_cmap = 'copper'):
        # Extract bounds from self.DEM
        buffer = 0.02
        min_lon = self.dem.lon.min().item()-buffer
        max_lon = self.dem.lon.max().item()+buffer
        min_lat = self.dem.lat.min().item()-buffer
        max_lat = self.dem.lat.max().item()+buffer
        
        bounds_polygon = {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
                [min_lon, max_lat]
            ]]
        }

        # folder = "/home/pwiersma/scratch/Data/GIS"
        # file = join(folder, "swissalti3d_2021_2570-1199_0.5_2056_5728.tif")
        
        folder = "/home/pwiersma/scratch/Data/DEM/DHM25_MM_ASCII_GRID/ASCII_GRID_1part"
        file = join(folder, "dhm25_grid_epsg4326.tif")

        with rasterio.open(file) as src:
            # Clip the raster data using the defined bounds
            out_image, out_transform = mask(src, [bounds_polygon], 
                                            crop=True,nodata= 0)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})

            # Read the clipped raster data
            data = out_image[0]

            # Manually remove the left and right columns and the bottom row
            data = data[:, 1:-1]  # Remove first and last columns
            data = data[:-1, :]   # Remove last row

            # Generate hillshade
            ls = LightSource(azdeg=315, altdeg=45)
            hillshade = ls.hillshade(data, vert_exag=1, dx=1, dy=1)

            # Print metadata
            print(out_meta)

            # Plot the data using imshow
            img = ax.imshow(data, cmap=dem_cmap, alpha=0.9, extent=(
                min_lon, max_lon, min_lat, max_lat))
            ax.imshow(hillshade, cmap='gray', alpha=0.25, extent=(
                min_lon, max_lon, min_lat, max_lat))
            cbar = plt.colorbar(img, ax=ax, label='Elevation')
            ax.set_title('Clipped DEM Data with Hillshade')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            # plt.colorbar(img, ax=ax, label='Elevation')
            return ax
    def create_dhm25_dem(self,output_path):
        if not hasattr(self,'dem'):
            self.dem =  xr.open_dataset(join(self.ROOTDIR,'wflow_staticmaps',f'staticmaps_{self.RESOLUTION}_{self.BASIN}_feb2024.nc'))['wflow_dem']

        buffer = 0.02
        min_lon = self.dem.lon.min().item()-buffer
        max_lon = self.dem.lon.max().item()+buffer
        min_lat = self.dem.lat.min().item()-buffer
        max_lat = self.dem.lat.max().item()+buffer
        
        bounds_polygon = {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
                [min_lon, max_lat]
            ]]
        }

        # folder = "/home/pwiersma/scratch/Data/GIS"
        # file = join(folder, "swissalti3d_2021_2570-1199_0.5_2056_5728.tif")
        
        folder = "/home/pwiersma/scratch/Data/DEM/DHM25_MM_ASCII_GRID/ASCII_GRID_1part"
        file = join(folder, "dhm25_grid_epsg4326.tif")

        with rasterio.open(file) as src:
            # Clip the raster data using the defined bounds
            out_image, out_transform = mask(src, [bounds_polygon], 
                                            crop=True,nodata= 0)
            
            #remove the left column and the bottom row from out_imgae
            out_image = out_image[:,:-1,1:]
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})
            
            
            # out_image = out_image[:-1,:,:]

            #

            #Write out_meta to a tif file 
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
            
        self.load_dhm25_dem()

           
    def load_dhm25_dem(self):
            
        dhm25_path = join(self.ROOTDIR,"aux_data","dhm25_clipped",f"{self.BASIN}_dhm25.tif")
        self.dhm25_path = dhm25_path
        if os.path.exists(dhm25_path):
            dhm25 = rioxr.open_rasterio(dhm25_path).sel(band = 1).squeeze()
            #remove "band" and "spatial_ref" coordinates
            dhm25 = dhm25.drop_vars(['band','spatial_ref'])
            dhm25 = dhm25.rename({'x':'lon','y':'lat'})
            self.dhm25 =    dhm25 
        else:
            self.create_dhm25_dem(dhm25_path)

    
            

    def plot_delin_on_ax(ax):
        folder = "/home/pwiersma/scratch/Data/Basins"
        directories = os.listdir(folder)
        for directory in directories:
            if self.BASIN.title() in directory:
                if not 'zip' in directory:
                    number = directory[3:7]
                    file  = gpd.read_file(join(folder,directory,f"CH-{number}.shp"))
        self.delin = file
        file.plot(ax = ax, color = 'none', edgecolor = 'black', linewidth = 0.5)
        S = SwissStation(self.BASIN)
        ax.plot(S.lon,S.lat, 'ro',label = 'Station')
        
    
    def plot_rivers(ax):
        clipped_dir = "/home/pwiersma/scratch/Data/ewatercycle/aux_data/clipped_rivers"
        clipped_file = join(clipped_dir,f"{self.BASIN}_rivers.shp")
        if os.path.exists(clipped_file):
            rivers_dem = gpd.read_file(clipped_file)
        else:
            rivers_folder = "/home/pwiersma/scratch/Data/GIS/Hydrography/FlussordnungszahlStrahler"
            # rivers_folder = "/home/pwiersma/scratch/Data/HydroMT"
            rivers_file = join(rivers_folder,"FLOZ_epsg4326.shp")
            # rivers_file = join(rivers_folder,"gdf_riv_hydroatlas.shp")

            rivers = gpd.read_file(rivers_file)

            rivers_dem = gpd.clip(rivers, self.delin.geometry)
            rivers_dem.to_file(clipped_file)
        # Plot the rivers
        rivers_dem.plot(ax = ax, color='blue', linewidth=1, alpha = 0.5)


    def plot_fancydem():
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_dem_with_hillshade(ax,dem_cmap = 'copper')
        plot_delin_on_ax(ax)
        plot_rivers(ax)
        # plot_SWE_as_ax(ax)
        plt.show()


# OSHD = C.load_OSHD(resolution = self.RESOLUTION).rename({'x':'lon','y':'lat'})
# OSHD = xr.where(np.isnan(self.dem),np.nan,OSHD)
# # for x in range(0,OSHD.x.size):
# #     for y in range(0,OSHD.y.size):
# #         OSHD_p = OSHD.isel(x = x, y = y)
# #         runoff = OSHD_p['romc_all'].to_series()
# #         OSHD_pdiff =  OSHD_p['swee_all'].diff(dim = 'time').to_series()
# #         melt = OSHD_pdiff.where(OSHD_pdiff <0)
# #         snowfall = OSHD_pdiff.where(OSHD_pdiff >0)
# #         plt.figure()
# #         plt.plot(runoff.cumsum())
# #         plt.plot(melt.cumsum()*-1)
# #         plt.plot(snowfall.cumsum())
# #         plt.title(f"{x},{y}")

# SWEdif = OSHD['swee_all'].diff(dim = 'time')
# snowfall = SWEdif.where(SWEdif > 0).sum(dim = ('lon','lat'))
# melt = SWEdif.where(SWEdif < 0).sum(dim = ('lon','lat'))
# # runoff = OSHD['romc_all'].sum(dim = ('lon','lat'))
# runoff = OSHD['romc_all'].mean(dim = ('lon','lat'))

# plt.figure()
# plt.plot(runoff.cumsum())
# plt.plot(melt.cumsum()*-1)

