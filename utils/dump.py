#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:35:42 2023

@author: pwiersma
"""

### Script to compare different masswasting settings
SWE_diff = SWE_list[2] - SWE_list[0]
monthly_SWE_diff = SWE_diff.groupby("time.month").mean()
# yearly_SWE_diff = SWE_diff.groupby("time.year").mean()

for i in range(2,4):
    monthly_SWE_diff.isel(month = i).plot(cmap = 'RdBu_r',vmin = -250,vmax = 250)
    for j,station in enumerate(C.station_short_names):
        # lon,lat = C.station_coordinates[station]
        lon,lat = C.HS_obs[station].attrs['coordinates']
        plt.scatter(lon,lat, color = 'black',marker = 'x', label = 'Snow station')
        if j ==0:
            plt.legend()
    plt.title(f'Difference from masswasting in month {i+1}')
    plt.show()
    
# SWE_True = SWE_list[0]
# monthly_SWE_True = SWE_True.groupby('time.month').mean()
# for i in range(0,6):
#     monthly_SWE_True.isel(month = i).plot()
#     for j,station in enumerate(C.station_short_names):
#         # lon,lat = C.station_coordinates[station]
#         lon,lat = C.HS_obs[station].attrs['coordinates']
#         plt.scatter(lon,lat, color = 'black',marker = 'x', label = 'Snow station')
#         if j ==0:
#             plt.legend()
#     plt.title(f'SWE in month {i+1}')
#     plt.show()
# SWE_diff.mean(dim = 'time').plot(cmap = "RdBu_r")
   
#%%import os
import glob
import shutil

# Get the base directory from an environment variable
# base_dir = os.getenv('EWATERCYCLE_BASE_DIR', '/default/path/to/ewatercycle')
base_dir = '/home/pwiersma/scratch/Data/ewatercycle'
base_dir = "/work/FAC/FGSE/IDYST/gmariet1/gaia/pwiersma/ewatercycle"

# Define the paths
input_path = os.path.join(base_dir, 'experiments/data/input/*')
data_path = os.path.join(base_dir, 'experiments/data/')
config_path = os.path.join(base_dir, 'experiments/*toml')

# Cleanup input files
files = glob.glob(input_path)
print(f"Found {len(files)} input files.")
for i, f in enumerate(files):
    if len(os.path.basename(f)) > 60 or 'tt_scale' in os.path.basename(f):
        os.remove(f)
    if i % 1000 == 0:
        print(f"Processed {i} input files.")

# Cleanup folders
folders = os.listdir(data_path)
print(f"Found {len(folders)} folders.")
for i, f in enumerate(folders):
    if 'clusters_m' in f or 'test' in f or 'sample' in f:
        try:
            shutil.rmtree(os.path.join(data_path, f))
        except Exception as e:
            print(f"{f} couldn't be removed: {e}")
            continue
    if i % 100 == 0:
        print(f"Processed {i} folders.")

# Cleanup config files
files = glob.glob(config_path)
print(f"Found {len(files)} config files.")
for i, f in enumerate(files):
    if 'clusters_posterior' in f or 'test' in f:
        os.remove(f)
    if i % 1000 == 0:
        print(f"Processed {i} config files.")

containers = os.listdir(base_dir+'/experiments/containers')
for dir in containers:
    files = glob.glob(base_dir+'/experiments/containers/'+dir+'/*')
    for file in files:
        if 'prior' in file or 'test' in file or 'posterior' in file in 'sample' in file:
            os.remove(file)

ewc_outputfolders = os.listdir(base_dir+'/experiments/ewc_outputs')
for dir in ewc_outputfolders:
    try:
        shutil.rmtree(dir)
    except Exception as e:
        print(f"{f} couldn't be removed: {e}")
        continue
    if i % 100 == 0:
        print(f"Processed {i} folders.")


#%%

###Script to develop the DOY benchmark 
    # WFJ = C.HSSWE['5WJ0']
    # WFJ_SWE = WFJ['SWE_jonas2009']
    # WFJ_DOY = WFJ_SWE.groupby(WFJ_SWE.index.day_of_year).mean()
    # WFJ_snowfallmeans = WFJ.groupby('hyear').sum()['HN_1D']*10
    # WFJ_snowfallmean = WFJ_snowfallmeans.mean()
    # for year in range(2000,2007):
    #     f1,ax1 = plt.subplots(figsize=(10,5))
    #     timeslice = slice(f"{year-1}-10",f"{year}-09")
    #     SWE = C.HSSWE[station]['SWE_jonas2009'][timeslice]
    #     SWE.plot(ax = ax1,label = 'jonas2009')
    #     index_DOY = SWE.index.day_of_year
    #     # if station in C.SWE_stations:
    #     #     (C.HSSWE[station]['HNW_1D'][timeslice]*10).plot(ax = ax1, label = 'Manual SWE obs',
    #     #                                                linestyle = '', marker = 'x',color = 'black')
    #     snowfall_days = np.sum(WFJ['HN_1D'][timeslice]>0)
    #     mean_diff = (WFJ_snowfallmeans[year] - WFJ_snowfallmean) /snowfall_days
        
    #     ratio = WFJ_snowfallmeans[year] / WFJ_snowfallmean
    #     reduction = WFJ_DOY.max() *(1-ratio)
        
    #     reduction = WFJ_DOY.mean() - SWE.mean()
        
    #     WFJ_DOY_reduced = WFJ_DOY -reduction
    #     WFJ_DOY_reduced = WFJ_DOY_reduced[index_DOY]
    #     # WFJ_DOY_reduced.index = SWE.index
    #     # WFJ_DOY_reduced.iloc[(SWE==0).tolist()] = 0
        
    #     ax1.plot(SWE.index,WFJ_DOY[index_DOY] * (WFJ_snowfallmeans[year] / WFJ_snowfallmean), label = 'DOYmean_normalized')
    #     ax1.plot(SWE.index,WFJ_DOY[index_DOY],label = 'DOYmean')
    #     ax1.plot(SWE.index,WFJ_DOY_reduced, label = 'DOYmean_reduced')
            
            
        # date1 = dt.datetime(year, 10, 1)
        # date2 = dt.datetime(1970, 1, 1)
        
        # # Calculate the timedelta
        # days_no = C.HSSWE[station][timeslice].index.size
        # x0 = (date1 - date2).days
        # x = x0 + WFJ_DOY.index
        # ax1.plot(x[:(366-275)],WFJ_DOY[275:])
        # ax1.plot(x[(366-183):], WFJ_DOY[:183])
        
        
        # 183
        
        # ax1.plot(WFJ_DOY)

        # ax1.set_title(f"{station} SWE evaluation")
        # ax1.legend()
        # ax1.grid()
        # ax1.set_ylabel('SWE jonas2009[mm]')
        # plt.show()
        
    from rasterio.enums import Resampling
    def convert_time_string_to_datetime(time_string):
        try:
            # Parse the time string using the specified format
            time_format = "%Y%m%d%H%M"
            datetime_obj = datetime.strptime(time_string, time_format)
            return datetime_obj
        except ValueError:
            # Handle invalid input gracefully
            return None
    OSHD = xr.open_dataset("/home/pwiersma/scratch/Data/SLF/OSHD/TI_1998_2022_OSHD_1km.nc", decode_times = False)
    OSHD = OSHD.rename(dict(xx = 'x', yy = 'y'))
    OSHD = OSHD.rio.write_crs('epsg:21781')
    OSHD = OSHD.rio.reproject(dst_crs = 'epsg:4326',resolution = 0.00833333, resampling = Resampling.bilinear)
    dt_time = [convert_time_string_to_datetime(str(int(t))) for t in OSHD.time.data]
    # OSHD = OSHD.rename(dict(x = 'lon',y = 'lat'))
    OSHD['time'] = dt_time
    del OSHD.romc_all.attrs['grid_mapping']
    del OSHD.dem.attrs['grid_mapping']
    del OSHD.swee_all.attrs['grid_mapping']
    del OSHD.forest.attrs['grid_mapping']
    del OSHD.open.attrs['grid_mapping']
    OSHD.to_netcdf("/home/pwiersma/scratch/Data/SLF/OSHD_1km_bilinear.nc")
    # OSHD = xr.where(OSHD == -9999, np.nan, OSHD)
    
    landuse = xr.open_dataset("/home/pwiersma/scratch/Scripts/landuse.nc")
    landuse = landuse.assign_coords(dict(ncols = OSHD['x'].data,nrows = OSHD['y'].data)).rename(dict(ncols = 'xx',nrows = 'yy'))
    landuse = landuse.transpose('yy','xx')
    landuse.attrs['cellsize'] = 1000
    landuse.attrs['NODATA_value'] = 'NaN'
    landuse.attrs['crs'] = 'EPSG:21781 / CH1903'
    landuse.to_netcdf("/home/pwiersma/scratch/Data/SLF/landuse_SLF_CH1903_1km.nc") 
    
    
    #%% OSHD 250m 
    for year in range(2016,2022):
        print(year)
        fsm = xr.open_dataset(f"/home/pwiersma/scratch/Data/SLF/envidat_download/250m/FSM2oshd_{year}.nc"
                              , decode_times = False, chunks = 'auto')
        varnames = list(fsm.data_vars)
        selected_vars = ['swet_all','dem','romc_all','forest']
        fsm = fsm[selected_vars]
        # fsm = fsm.isel(time = slice(0,5))
        fsm = fsm.rename(dict(xx = 'x', yy = 'y'))
        fsm = fsm.rio.write_crs('epsg:21781')
        fsm = fsm.rio.reproject(dst_crs = 'epsg:4326',resolution = 0.0020833325, resampling = Resampling.bilinear)
        dt_time = [convert_time_string_to_datetime(str(int(t))) for t in fsm.time.data]
        # fsm = fsm.rename(dict(x = 'lon',y = 'lat'))
        fsm['time'] = dt_time
        del fsm.romc_all.attrs['grid_mapping']
        del fsm.dem.attrs['grid_mapping']
        del fsm.swet_all.attrs['grid_mapping']
        del fsm.forest.attrs['grid_mapping']
        fsm.to_netcdf(f"/home/pwiersma/scratch/Data/SLF/FSM/FSM_250m_bilinear_{year}.nc")
#%%
    basins = ['Mogelsberg','Dischma','Riale_di_Pincascia','Ova_da_Cluozza','Sitter','Kleine_Emme','Werthenstein','Sense',
              'Sense','Ilfis','Eggiwil','Chamuerabach','Veveyse','Minster','Ova_dal_Fuorn','Alp','Biber','Muota','Riale_di_Calneggia',
              'Chli_Schliere','Allenbach']
    for basin in basins:
        print(basin)
        fsm500 = xr.open_dataset(f"/home/pwiersma/scratch/Data/SLF/FSM/FSM_500m_latlon_{basin}.nc")
        dem1000 = xr.open_dataset(f"/home/pwiersma/scratch/Data/ewatercycle/wflow_staticmaps/staticmaps_1000m_{basin}_feb2024.nc")['wflow_dem']
        
        fsm500 = fsm500.rio.write_crs('epsg:4326')
        # fsm500 = fsm500.transpose('time','lat','lon')
        dem1000 = dem1000.rio.write_crs('epsg:4326').rename(dict(lon = 'x',lat = 'y'))
              
        fsm1000 = fsm500.rio.reproject_match(dem1000)
        
        del fsm1000.romc_all.attrs['grid_mapping']
        del fsm1000.dem.attrs['grid_mapping']
        del fsm1000.swet_all.attrs['grid_mapping']
        del fsm1000.forest.attrs['grid_mapping']
        fsm1000.to_netcdf(f"/home/pwiersma/scratch/Data/SLF/FSM/FSM_1000m_latlon_{basin}.nc")

    #%%
    SWEmax = xr.open_dataset(f"/home/pwiersma/scratch/Data/SLF/OSHD_1km_latlon_{basin}_yearlymax.nc")['swee_all']
    SWE = SWEmax[0]*20

    # SWEmax = 300
    # SWE = 150
    CV = 0.5

    ksi2 = np.log1p(1+CV**2)
    labda = np.log1p(SWEmax) - 0.5 * ksi2
    fsca = (1/(SWE*np.sqrt(ksi2)*np.sqrt(2*np.pi))) *np.exp(-0.5 * ((np.log1p(SWE)-labda)/np.sqrt(ksi2))**2)

    fsca = np.tanh(1.26 * (SWE/(CV * SWEmax)))

    def fsca(SWE, SWEmax, CV):
        return np.tanh(1.26 * (SWE/(CV * SWEmax)))

#%%
    
#%% Convert RhiresD folder = "/home/pwiersma/scratch/Data/MeteoSwiss/RhiresD_2001_23ch01h.swiss.lv95" files = glob.glob(join(folder,"*")) 
#ch02 = xr.open_dataset("/home/pwiersma/scratch/Data/MeteoSwiss/RhiresD_1961_2017_ch02_lonlat/RhiresD_ch02.lonlat_201001010000_201012310000.nc") 
#ch02 = ch02.sel(time = '2010-01-01')['RhiresD'].rio.write_crs('epsg:4326') for f in files: # file = xr.open_dataset(join(folder,"RhiresD_ch01h.swiss.lv95_201001010000_201012310000.nc"))
# file = xr.open_dataset(f) # file = file.sel(time = '2010-01-01')['RhiresD'] file = file.rename(dict(E = 'x',N = 'y')) file = file.drop_vars(['lat','lon']).rio.write_crs('epsg:2056')

# Define the output folder
output_folder = "/home/pwiersma/scratch/Data/MeteoSwiss/RhiresD_ch01_to_ch02_latlon"

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

folder = "/home/pwiersma/scratch/Data/MeteoSwiss/RhiresD_81_2000.ch01h.swiss.lv95"
files = glob.glob(join(folder,"*"))
ch02 = xr.open_dataset("/home/pwiersma/scratch/Data/MeteoSwiss/RhiresD_1961_2017_ch02_lonlat/RhiresD_ch02.lonlat_201001010000_201012310000.nc")
ch02 = ch02.sel(time = '2010-01-01')['RhiresD'].rio.write_crs('epsg:4326')

for f in files:
    print(f)
    file = xr.open_dataset(f)
    file = file.rename(dict(E = 'x',N = 'y'))
    file = file.drop_vars(['lat','lon']).rio.write_crs('epsg:2056')
    
    new = file.rio.reproject_match(ch02)

    # Extract the year from the filename
    year = os.path.basename(f).split("_")[2][:4]

    # Define the output filename
    output_filename = f"RhiresD_ch01_to_ch02_{year}0101_{year}1231.nc"

    # Save the reprojected dataset
    new = new.drop_vars('swiss_lv95_coordinates')
    del new['RhiresD'].attrs['grid_mapping']
    new.to_netcdf(join(output_folder, output_filename))


    #%%
    # for year in range(2005,2010):
    #     obs = evaldic[basin][year]['Q']['obs']
    #     test = thresholding_algo(obs.values,lag = 5,threshold = 5,influence = 1)
    #     test2 = thresholding_algo(obs.values,lag = 100,threshold = 3,influence = 1)
        
    #     diff = np.diff(obs,prepend = 0)
    #     posdif = (diff>0) *diff
    #     std = obs.std()
    #     # mask = posdif>np.quantile(posdif,0.9)
    #     mask1 = posdif>5*posdif.std()
        
    #     mask2 = obs.values>np.quantile(obs.values,0.99)
    #     mask = mask1 | mask2 
        
        
    #     plt.figure(figsize = (15,5))
    #     #plt.plot(test['avgFilter'])
    #     plt.plot(obs.values)
    #     filtered = np.where(test['signals']==1,np.nan,obs.values)
    #     filtered2 = np.where(test2['signals'] ==1, np.nan, obs.values)
    #     filtered3 = np.where(mask,np.nan,obs.values)
    #     # plt.plot(filtered,label = 'treshold = 5')
    #     # plt.plot(filtered2,label = 'treshold = 3')
    #     plt.plot(filtered3,linestyle = None, marker = 'o')
    #     plt.legend()
#%%
# for basin in ['Jonschwil','Landquart','Landwasser','Rom','Verzasca']:
# for basin in ['Mogelsberg','Dischma','Riale_di_Pincascia','Ova_da_Cluozza','Sitter','Kleine_Emme','Werthenstein','Sense',
#               'Sense','Ilfis','Eggiwil','Chamuerabach','Veveyse','Minster','Ova_dal_Fuorn','Alp','Biber','Muota','Riale_di_Calneggia',
#               'Chli_Schliere','Allenbach']:
    basin  = 'Riale_di_Calneggia'
    resolution = '50m'
    path = "/home/pwiersma/scratch/Data/ewatercycle/wflow_staticmaps/"
    # old_nc = join(path,f"staticmaps_1000m_{basin}_aug2023.nc")
    new_nc = join(path,f"staticmaps_{resolution}_{basin}_feb2024.nc")


    old_nc = join(f"/home/pwiersma/scratch/Data/HydroMT/model_builds/wflow_{basin}_{resolution}_feb2024/staticmaps.nc")
    

    sm1 = xr.open_dataset(old_nc)
    sm1['rfcf'] = xr.where(np.isnan(sm1['wflow_ldd']),np.nan,1)
    sm1['sfcf'] = xr.where(np.isnan(sm1['wflow_ldd']),np.nan,1)
    sm1['vegcf'] = xr.where(np.isnan(sm1['wflow_ldd']),np.nan,0)
    sm1['CV'] = xr.where(np.isnan(sm1['wflow_ldd']),np.nan,0)
    sm1['mwf'] = xr.where(np.isnan(sm1['wflow_ldd']),np.nan,0)
    sm1.to_netcdf(new_nc)

#%%

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create x, y coordinates
x = sm.lon.data
y = sm.lat.data
x, y = np.meshgrid(x,y)

# # Create a bar for each grid cell
# for i in range(x.shape[0]):
#     for j in range(x.shape[1]):
#         ax.bar3d(x[i, j], y[i, j], 0, 1, 1, sm['wflow_dem'].data[i, j], shade=True)
var = sm['wflow_ldd']
norm = plt.Normalize(var.min(), var.max())

# Create a colormap
colors = cm.viridis(norm(var))

rcount, ccount, _ = colors.shape

p = ax.plot_surface(x,y, sm['wflow_dem'].data,cmap='terrain',shade = False)
# p = ax.plot_surface(x,y, sm['wflow_dem'].data,
#                 facecolors = colors,shade = False)
ax.view_init(elev=30, azim=270)
ax.grid(False)
ax.set_zlim(1500,7000)
fig.colorbar(p)
# ax.axis('off')
plt.show()

#%%
import glob
import os
files = glob.glob("/home/pwiersma/scratch/Data/ewatercycle/experiments/data/input/*")
for i,f in enumerate(files):
    if len(f)>150:
        print(f[-30:])
        os.remove(f)
    if i%1000==0:
        print(i)
        
        
#%% OSHD lapse rate analysis
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
for basin in ['Landquart','Landwasser','Verzasca','Jonschwil','Rom']:
    OSHD = xr.open_dataset(f"/home/pwiersma/scratch/Data/SLF/OSHD_1km_latlon_{basin}.nc").rename(dict(x = 'lon',y = 'lat'))
    SWE = OSHD.swee_all.sel(time = OSHD.time.dt.month ==1).sel(time = slice('2004','2015'))
    T = xr.open_dataset(f"/home/pwiersma/scratch/Data/ewatercycle/wflow_Julia_forcing/wflow_MeteoSwiss_1000m_{basin}_2004_2015.nc").tas
    T = T.sel(time = T.time.dt.month ==1)
    
    
    dem = OSHD.dem
    
    SWE = xr.where(dem<1000,np.nan,SWE)
    T = xr.where(dem<1000,np.nan,T)
    
    time = SWE.time.to_pandas().index
    timemask = time[T.max(dim = ['lat','lon'])<0]
    
    SWE = SWE.sel(time = timemask)
    
    # Calculate the snowfall from the SWE data
    snowfall = SWE.diff(dim='time').clip(min=0)
    
    # Create elevation bands
    elevation_bands = (dem / 100).round() * 100
    
    # Group the snowfall data by elevation band
    snowfall_grouped_by_elevation = snowfall.groupby(elevation_bands)
    
    # Calculate the mean snowfall for each elevation band and time step
    sf = snowfall_grouped_by_elevation.mean()
    time_mask = sf.sum(dim ='dem')>100
    sf = sf.sel(time = sf.time[time_mask]).transpose()
    # sf = xr.where(sf ==0,np.nan,sf)
    
    f1,(ax1,ax2,ax3) = plt.subplots(1,3,figsize = (20,5))
    slopes = []
    rvalues = []
    for t in sf.time:
        event = sf.sel(time =t).isel(dem = slice(6,26))#.to_dataframe(name = 'snowfall')
        slope,intercept,r,p,se = ss.linregress(event.dem.data,event.data)
        slopes.append(slope)
        rvalues.append(r)

        if slope<-0.005:
            color = 'tab:cyan'
            alpha = 0.15
        elif slope>0.005:
            color = 'tab:red'
            alpha = 0.15
        else:
            color = 'black'
            alpha = 0.05
        event.plot(y = 'dem', color = color,alpha = alpha,ax = ax1)
    ax1.set_xlim(0,50)
    ax1.set_xlabel('Snowfall rate [mm/day]')
    ax1.set_title(f'{basin} Snowfall events (sum>10mm,Tmax=10 degC)')
    
    slopes = np.array(slopes)
    rvalues = np.array(rvalues)
    # plt.figure()
    # plt.plot(slopes)
    
    sns.histplot(slopes*1000,ax = ax2)
    ax2.set_title(f'{basin} snowfall lapse rates')
    ax2.set_xlabel('Lapse rate [mm/km]')
    ax2.set_xlim(-20,60)
    ax2.axvline(0,linestyle = 'dashed', color = 'black')
    
    sns.histplot(rvalues**2,ax = ax3)
    ax3.set_title(f'{basin} R2 of fits')
    ax3.set_xlabel('R2')
    # ax3.set_xlim(-20,60)
    # ax3.axvline(0,linestyle = 'dashed', color = 'black')
    
    f1.savefig(f"/home/pwiersma/scratch/Figures/ewc_figures/snowfall_lapse_rates/{basin}.png",dpi = 300)
    
    # sns.lmplot(data = event, x = 'snowfall',y = 'dem')
    




# sf.plot(figsize = (15,5),vmin = 0,vmax = 50)

#%%
snowfall_flat = snowfall.data.flatten()
elevation_flat = np.tile(OSHD.dem.data.flatten(),snowfall.time.data.size)

zeromask =  snowfall_flat > 1

sfmasked = snowfall_flat[zeromask]
z = elevation_flat[zeromask]

limit = 100000
# plt.scatter(elevation_flat,snowfall_flat, s = 0.05)
sns.kdeplot(x = z[:limit],y = sfmasked[:limit], fill = True, alpha = 1)
sns.kdeplot(x = z[:limit],y = sfmasked[:limit], alpha = 1)

sns.scatterplot(x = z[:limit],y = sfmasked[:limit],s = 0.5, alpha = 0.5, color = 'black')

plt.xlabel('Elevation [m]')
plt.ylabel('Snowfall [mm]')
plt.title('Landquart snowfall over elevation')
plt.ylim(0,100)

#%%
# Flatten the snowfall and elevation data
snowfall_flat = snowfall.stack(z=('lat', 'lon', 'time'))
elevation_flat = dem.stack(z=('lat', 'lon'))

# Align the snowfall and elevation data
snowfall_flat, elevation_flat = xr.align(snowfall_flat, elevation_flat, join='inner')

# Convert to pandas DataFrame for easier analysis
df = pd.DataFrame({
    'snowfall': snowfall_flat.to_pandas(),
    'elevation': elevation_flat.to_pandas()
})

# Drop rows with missing data
df = df.dropna()

# # Perform a correlation analysis
# correlation = df['snowfall'].corr(df['elevation'])

# print(correlation)'Muota',
#%%
# for basin in ['Allenbach']:
basins =   ['Mogelsberg','Dischma','Ova_da_Cluozza','Sitter','Werthenstein',
                  'Sense','Ilfis','Eggiwil','Chamuerabach','Veveyse','Minster','Ova_dal_Fuorn','Alp','Biber','Riale_di_Calneggia',
                  'Chli_Schliere','Allenbach','Kleine_Emme','Riale_di_Pincascia','Jonschwil','Landwasser','Landquart','Rom','Verzasca']
pet_comparison = pd.DataFrame(index = basins)
for basin in  basins:
    forcing_name = join("/home/pwiersma/scratch/Data/ewatercycle/wflow_Julia_forcing",f"wflow_MeteoSwiss_500m_{basin}_1993_2019.nc")
    generate_MS_forcing(forcing_name,basin = basin, resolution = '500m',start_year = 1993,end_year = 2017)
    forcing = xr.open_dataset(forcing_name)
    pet = forcing.pet.mean(dim = ['lat','lon']).resample(time = 'Y').sum().to_pandas()
    
    station_number = SwissStation(basin).number
    CAMELS_simpath = f"/home/pwiersma/scratch/Data/CAMELS/CAMELS-CH/camels_ch/time_series/simulation_based/CAMELS_CH_sim_based_{station_number}.csv"
    sim = pd.read_csv(CAMELS_simpath,parse_dates = True,
                      index_col = 0,delimiter = ';')
    prevah = sim['pet_sim(mm/d)']
    prevah = prevah.loc[slice('1993','2019')]
    prevah = prevah.resample('Y').sum()
    
    f1,ax1 = plt.subplots()
    prevah.plot(ax = ax1,label = 'PREVAH PET (CAMELS)')
    pet.plot(ax = ax1, label = 'DeBruin PET (ESMVALTOOL)')
    ax1.legend()
    ax1.grid()
    ax1.set_title(f"{basin} ET comparison")
    # pet_comparison[basin] = pet.mean()/prevah.mean()
    pet_comparison.loc[basin,'DeBruin']  = pet.mean()
    pet_comparison.loc[basin,'PREVAH'] = prevah.mean()
    pet_comparison.loc[basin,'ratio'] = pet.mean()/prevah.mean()
# pet_comparison.to_csv("/home/pwiersma/scratch/Data/ewatercycle/pet_comparison.csv")
    
#%%
# for basin in ['Allenbach']:
# basins = ['Allenbach']
basins =   ['Mogelsberg','Dischma','Ova_da_Cluozza','Sitter','Werthenstein',
                  'Sense','Ilfis','Eggiwil','Chamuerabach','Veveyse','Minster','Ova_dal_Fuorn','Alp','Biber','Riale_di_Calneggia',
                  'Chli_Schliere','Allenbach','Kleine_Emme','Riale_di_Pincascia','Jonschwil','Landwasser','Landquart','Rom','Verzasca']
pet_comparison = pd.DataFrame(index = basins)
for basin in  basins:
    forcing_name = join("/home/pwiersma/scratch/Data/ewatercycle/wflow_Julia_forcing",f"wflow_MeteoSwiss_1000m_{basin}_1993_2019.nc")
    generate_MS_forcing(forcing_name,basin = basin, resolution = '1000m',start_year = 1993,end_year = 2019)
    forcing = xr.open_dataset(forcing_name)
    pet = forcing.pet.mean(dim = ['lat','lon']).resample(time = 'M').sum().to_pandas()
    
    station_number = SwissStation(basin).number
    CAMELS_simpath = f"/home/pwiersma/scratch/Data/CAMELS/CAMELS-CH/camels_ch/time_series/simulation_based/CAMELS_CH_sim_based_{station_number}.csv"
    sim = pd.read_csv(CAMELS_simpath,parse_dates = True,
                      index_col = 0,delimiter = ';')
    prevah = sim['pet_sim(mm/d)']
    prevah = prevah.loc[slice('1993','2019')]
    prevah = prevah.resample('M').sum()
    
    f1,ax1 = plt.subplots()
    prevah.plot(ax = ax1,label = 'PREVAH PET (CAMELS)')
    pet.plot(ax = ax1, label = 'DeBruin PET (ESMVALTOOL)')
    ax1.legend()
    ax1.grid()
    ax1.set_title(f"{basin} ET comparison")
    
    ratio = prevah/pet
    plt.figure()
    ratio.plot()    
    
    monthly_ratio = ratio.groupby(ratio.index.month).mean()
    plt.figure()
    monthly_ratio.plot()
    for i in range(1,13):
        pet_comparison.loc[basin,i] = monthly_ratio[i]
    
    
    # pet_comparison[basin] = pet.mean()/prevah.mean()
    pet_comparison.loc[basin,'DeBruin']  = pet.mean()
    pet_comparison.loc[basin,'PREVAH'] = prevah.mean()
    pet_comparison.loc[basin,'ratio'] = prevah.mean()/pet.mean()
pet_comparison.to_csv("/home/pwiersma/scratch/Data/ewatercycle/pet_comparison_monthly.csv")


#Class to make dictionaries subscriptable
class DotDict:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def __getattr__(self, attr):
        return self.__dict__.get(attr)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, item):
        del self.__dict__[item]


import numpy as np
from matplotlib.colors import LightSource
from datetime import datetime, timedelta

def compute_average_shading(latitude, elevation_data):
    """
    Compute the average shading for every day of the year and the overall average shading for the entire year.
    
    Parameters:
    latitude (float): Latitude of the catchment.
    elevation_data (2D array): Elevation data of the catchment.
    
    Returns:
    2D array: Average shading for each grid cell for the entire year.
    """
    def solar_position(day_of_year, hour, latitude):
        # Calculate the solar declination angle
        declination = 23.44 * np.cos(np.deg2rad((360 / 365) * (day_of_year + 10)))
        
        # Calculate the hour angle
        hour_angle = 15 * (hour - 12)
        
        # Calculate the solar altitude angle
        altitude = np.arcsin(np.sin(np.deg2rad(latitude)) * np.sin(np.deg2rad(declination)) +
                             np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(declination)) * np.cos(np.deg2rad(hour_angle)))
        
        # Calculate the solar azimuth angle (zero being north)
        azimuth = np.arctan2(np.sin(np.deg2rad(hour_angle)), 
                             np.cos(np.deg2rad(hour_angle)) * np.sin(np.deg2rad(latitude)) - 
                             np.tan(np.deg2rad(declination)) * np.cos(np.deg2rad(latitude)))
        
        return np.rad2deg(azimuth), np.rad2deg(altitude)
    
    total_shading = np.zeros_like(elevation_data, dtype=float)
    days_in_year = 365
    
    for day in range(1, days_in_year + 1,50):
        daily_shading = np.zeros_like(elevation_data, dtype=float)
        hours_in_day = 24
        
        for hour in range(0,hours_in_day,6):
            azimuth, altitude = solar_position(day, hour, latitude)
            ls = LightSource(azdeg=azimuth, altdeg=altitude)

            if altitude > 0:  # Only consider times when the sun is above the horizon
                shading = ls.hillshade(elevation_data)
                plt.figure()
                plt.imshow(shading)
                plt.title(f"Day {day}, Hour {hour}")
                daily_shading += shading
        
        daily_shading /= hours_in_day
        total_shading += daily_shading
    
    average_shading = total_shading / days_in_year
    return average_shading

# Example usage:
latitude = 45.0  # Example latitude
elevation_data = dem.data  # Example elevation data
average_shading = compute_average_shading(latitude, elevation_data)
print(f"Average shading for the entire year (per grid cell):\n{average_shading}")





#%% Change name from Dischma to 


from hydrobricks import Catchment as hbc
import rioxarray as rioxr
import xarray as xr
from os.path import join
CC = hbc()
dem_path ='/home/pwiersma/scratch/Data/ewatercycle/aux_data/dhm25_clipped/Riale_di_Calneggia_dhm25.tif'
dem = rioxr.open_rasterio(dem_path).rio.write_crs('epsg:4326')
new_dem = dem.rio.reproject('epsg:21781')


CC.extract_dem(dem_path)
CC.calculate_slope_aspect()
output_path = "/home/pwiersma/scratch/Data/ewatercycle/aux_data/Ipot/"
CC.calculate_daily_potential_radiation(output_path = output_path, with_cast_shadows=False )
potrad = rioxr.open_rasterio(join(output_path,"annual_potential_radiation.tif"))
potrad.sel(band = 1).plot(vmin = 0)
    
    
daily_potrad = xr.open_dataset(join(output_path,"daily_potential_radiation.nc"))['radiation']
daily_potrad = daily_potrad.rio.write_crs('epsg:4326')
for d in [1,60,120,180,240,300]:
    plt.figure()
    dp = daily_potrad.sel(day_of_year = d)
    dp.plot(vmin = 0,vmax = 200)
    plt.title(dp.mean().item())
    plt.show()
daily_potrad.to_netcdf(join(output_path,"Ipot_DOY_Riale_di_Calneggia.nc"))

#resampling to grid 
resolution = '1000m'
dem = xr.open_dataset(f"/home/pwiersma/scratch/Data/ewatercycle/wflow_staticmaps/staticmaps_{resolution}_Riale_di_Calneggia_feb2024.nc")['wflow_dem']
dem = dem.rio.write_crs('epsg:4326')
daily_potrad_resampled = daily_potrad.rio.reproject_match(dem, resampling = Resampling.bilinear)
#remove negative values
daily_potrad_resampled = xr.where(daily_potrad_resampled<0,0,daily_potrad_resampled)
# daily_potrad_resampled = xr.where()
for d in [1,60,120,180,240,300]:
    plt.figure()
    dp = daily_potrad_resampled.sel(day_of_year = d)
    dp.plot()
    plt.title(dp.mean().item())
    plt.show()

daily_potrad_resampled.to_netcdf(join(output_path,f"Ipot_DOY_{resolution}_Riale_di_Calneggia.nc"))




#%% CHeck represnetativeness of 2014-2020
forcing_file = "/home/pwiersma/scratch/Data/ewatercycle/wflow_Julia_forcing/wflow_MeteoSwiss_1000m_Dischma_1961_2022.nc"
generate_MS_forcing(forcing_file,basin = 'Dischma', resolution = '1000m',start_year = 1961,end_year = 2022)

pr = xr.DataArray.from_iris(cubes[0])
tas = xr.DataArray.from_iris(cubes[1])
forcing= xr.Dataset({'pr':pr,'tas':tas})

mean_annual_temp = forcing.tas.resample(time = 'Y').mean().mean(dim = ['lat','lon']).compute()
mean_winter_temp = forcing.tas.sel(time = forcing.time.dt.month.isin([12,1,3])).resample(time = 'Y').mean().mean(dim = ['lat','lon']).compute()
mean_annual_precip = forcing.pr.resample(time = 'Y').sum().mean(dim = ['lat','lon']).compute()
mean_winter_precip = forcing.pr.sel(time = forcing.time.dt.month.isin([12,1,2,3])).resample(time = 'Y').sum().mean(dim = ['lat','lon']).compute()
#%%
f1,axes = plt.subplots(2,2,figsize = (15,10))
axes = axes.flatten()
axes[0].scatter(mean_annual_temp,mean_winter_temp)
axes[0].set_xlabel('Mean annual temperature [degC]')
axes[0].set_ylabel('Mean winter temperature [degC]')
axes[1].scatter(mean_annual_precip,mean_winter_precip)
axes[1].set_xlabel('Mean annual precipitation [mm]')
axes[1].set_ylabel('Mean winter precipitation [mm]')
axes[2].scatter(mean_annual_temp,mean_annual_precip)
axes[2].set_xlabel('Mean annual temperature [degC]')
axes[2].set_ylabel('Mean annual precipitation [mm]')
axes[3].scatter(mean_winter_temp,mean_winter_precip)
axes[3].set_xlabel('Mean winter temperature [degC]')
axes[3].set_ylabel('Mean winter precipitation [mm]')

#in the same figures, plot the 2014-2020 values
year0 = '1998'
year1 = '2022'
color = 'red'
marker = 'x'
markersize = 100
axes[0].scatter(mean_annual_temp.sel(time = slice(year0,year1)),
                mean_winter_temp.sel(time = slice(year0,year1)),
                        color = color,label = f"{year0}-{year1}",marker = marker,s = markersize)
axes[1].scatter(mean_annual_precip.sel(time = slice(year0,year1)),
                mean_winter_precip.sel(time = slice(year0,year1)),
                        color = color,label = f"{year0}-{year1}",marker = marker,s = markersize)
axes[2].scatter(mean_annual_temp.sel(time = slice(year0,year1)),
                mean_annual_precip.sel(time = slice(year0,year1)),
                        color = color,label = f"{year0}-{year1}",marker = marker,s = markersize)
axes[3].scatter(mean_winter_temp.sel(time = slice(year0,year1)),
                mean_winter_precip.sel(time = slice(year0,year1)),
                        color = color,label = f"{year0}-{year1}",marker = marker,s = markersize)

axes[0].legend()
plt.suptitle('Dischma meteo 1961-2022')


# %%
#%% Check out photogram maps 
folder = "/home/pwiersma/scratch/Data/SLF/photogrammetry/2m"
file = glob.glob(join(folder,"*.tiff"))[0]
data = xr.open_dataset(file,engine = 'rasterio').squeeze().to_dataarray()
data = data.rio.write_nodata(np.nan)

folder = "/home/pwiersma/scratch/Data/Basins"
directories = os.listdir(folder)
for directory in directories:
    if 'Dischma' in directory:
        if not 'zip' in directory:
            number = directory[3:7]
            delin  = gpd.read_file(join(folder,directory,f"CH-{number}.shp"))

dem =  xr.open_dataset(join("/home/pwiersma/scratch/Data/ewatercycle",
                       'wflow_staticmaps',
                       f'staticmaps_1000m_Dischma_feb2024.nc'))['wflow_dem']
dem = dem.rio.reproject('epsg:21781')
buffer = 0.1
min_lon = dem.x.min().item()-buffer
max_lon = dem.x.max().item()+buffer
min_lat = dem.y.min().item()-buffer
max_lat = dem.y.max().item()+buffer

bounds_polygon ={
    "type": "Polygon",
    "coordinates": [[
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat],
        [min_lon, min_lat],
        [min_lon, max_lat]
    ]]
}
bounds_polygon_str = json.dumps(bounds_polygon)
cropping_geometries = [geojson.loads(bounds_polygon_str)]

clipped =data.rio.clip(cropping_geometries)
clipped.to_netcdf(join(folder,"clipped_Dischma.tiff"))


with rasterio.open(file) as src:
    out_image, out_transform = mask(src, [bounds_polygon], 
                                        crop=True,nodata= 0)
    out_image = out_image[:,:-1,1:]
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})
    rioxr.


# %%

year = 2019
months = [4,5,6,7,8,9]
timeslice = slice(f"{year}-{months[0]}",f"{year}-{months[-1]}")
Q = wflow.stations['Dischma'].combined
obs = Q['obs'][timeslice]
keys = Q.columns
offset = par[1]-par[0]

diff = Q[keys[2]] - Q[keys[1]] 
# diff[diff<offset*0.1] = 0
rainfall_Q = diff / offset
# rainfall_Q[rainfall_Q<0] = 0
# rainfall_Q[rainfall_Q>Q[keys[2]]] = Q[keys[2]]

# f1,ax1 = plt.subplots(figsize = (10,5))
# obs.plot(ax = ax1, linestyle = 'dashed',color = 'black')
# for col in Q.columns:
#     if col =='obs':
#         continue
#     sim = Q[col][timeslice]
#     sim.plot(ax = ax1,label = col)

f1,ax1 = plt.subplots(figsize = (10,5))
# obs.plot(ax = ax1, linestyle = 'dashed',color = 'black')
Q[keys[2]][timeslice].plot(ax = ax1,label = "Total streamflow",color = 'black')
ax1.fill_between(x = rainfall_Q[timeslice].index,
                 y1 = rainfall_Q[timeslice]*0,
                  y2 = rainfall_Q[timeslice],
                  color = 'tab:blue',alpha = 1,
                 label = 'Snowmelt ')
ax1.fill_between(x = rainfall_Q[timeslice].index,
                    y1 = rainfall_Q[timeslice],
                    y2 = Q[keys[2]][timeslice],
                    color = 'tab:blue',alpha = 0.3,
                    label = 'Rainfall')

# rainfall_Q[timeslice].plot(ax = ax1,label = f"rainfall_Q",linestyle = 'dotted')

ax1.legend()
ax1.set_ylabel("Discharge [m3/s]")


from scipy.signal import savgol_filter
addition = snowfall_Q[timeslice] +rainfall_Q[timeslice]


ratio = snowfall_Q[timeslice]/(addition)
# ratio
#remove outliers
# new_ratio = ratio.rolling(window = 5, center = True).median()

from scipy.stats import zscore
z = zscore(ratio)
new_ratio = ratio.where(np.abs(z)<1,other = np.nan)
new_ratio[new_ratio<0] = 0
new_ratio[new_ratio>1] = 1
final_ratio  = new_ratio.interpolate(method = 'linear')

final_sf = final_ratio * Q[keys[2]][timeslice]
final_rf = (1-final_ratio) * Q[keys[2]][timeslice]



#%% Analyze outputs of Syn_249a
# folder = "/home/pwiersma/scratch/Data/ewatercycle/outputs/Syn_249a"

daterange = pd.date_range('2014-01-01','2014-08-30')
obs = E.SWE['Synthetic'].sel(time = daterange)
sim = E.SWE['posterior'].sel(time = daterange)
sim1 = sim['posterior_ms0_my0']

# obs = obs.sel(time = '2020-04-01')
elevation = E.dem
obs = xr.where(np.isnan(elevation),np.nan,obs)

# Flatten the arrays and remove NaN values
def flatten_and_clean(data, elevation):
    data_flat = data.data.flatten()
    elevation_flat = elevation.data.flatten()
    mask = ~np.isnan(data_flat) & ~np.isnan(elevation_flat)
    return data_flat[mask], elevation_flat[mask]
#%%
# months = np.unique(obs.time.dt.month)
months  = [11,12,1,2,3,4,5,6]
N = len(months)
f1,axes = plt.subplots(int(np.ceil(N/2)),2,figsize = (6,10),sharey = True,sharex = True)
axes = axes.flatten()
# plt.subplots_adjust(wspace = 0.5)
for i,m in enumerate(months): 
    ax = axes[i]
    days = obs.time[obs.time.dt.month == m]
    for ii,d in enumerate(days):
        # Prepare data
        for mo in range(2):
            if mo == 0:
                data = obs.sel(time = d)
                palette = sns.color_palette('Reds', len(days))
            else:
                data = sim1.sel(time = d)
                palette = sns.color_palette('Blues', len(days))
            snow_flat, elevation_flat = flatten_and_clean(data, elevation)

            # Create a DataFrame
            data = pd.DataFrame({'elevation': elevation_flat, 'snow': snow_flat})

            # Define bins
            bins = np.linspace(1700, 3000, 15)

            # Add a column indicating the bin for each elevation value
            data['elevation_bin'] = pd.cut(data['elevation'], bins)
            data['elevation_bin'] = pd.Categorical(data['elevation_bin'], ordered=True)
            binsum = data.groupby('elevation_bin',observed = False)['snow'].sum().reset_index()
            binsum['x'] = binsum['elevation_bin'].apply(lambda x: x.mid)
            ax.plot(binsum['snow'],binsum['x'], color = palette[ii],alpha = 0.5)
    ax.grid()
    ax.set_title(f"Month {m}")

#Plot how many grid cells are collaborating to the melt 
#%%
# obsdif = obs.diff(dim = 'time')
# obsmeltcount = xr.where(obsdif<0,1,0).sum(dim = ['lat','lon'])
# simdif = sim1.diff(dim = 'time')
# simmeltcount = xr.where(simdif<0,1,0).sum(dim = ['lat','lon'])

# obsmelt = self.daily_scalars['snowmelt']['Synthetic'].loc[obsdif.time]
# simmelt = self.daily_scalars['snowmelt']['posterior']['ms0_my0'].loc[simdif.time]

# obsratio = obsmelt/obsmeltcount
# simratio = simmelt/simmeltcount

# plt.figure()
# obsmeltcount.plot(label = 'obs')
# simmeltcount.plot(label = 'sim')
# plt.legend()

# f1,ax1 = plt.subplots(1,2,figsize = (8,3))
# obsmelt.plot(label = 'obs',ax = ax1[0])
# simmelt.plot(label = 'sim',ax = ax1[0])
# ax1[0].set_title('Snowmelt')
# plt.legend()

# obsratio.plot(label = 'obs',ax = ax1[1])
# simratio.plot(label = 'sim',ax = ax1[1])
# ax1[1].set_title('Snowmelt per grid cell')
# plt.legend()
# %%


#Find the difference between two xarray dataset 
path_static = "/home/pwiersma/scratch/Data/ewatercycle/experiments/containers/Syn_2310a/staticmaps_1000m_Dischma_feb2024_Soilcalib_Syn_2310a_Dischma_posterior_ms0.nc"
sm_post = xr.open_dataset(path_static)
path_static = "/home/pwiersma/scratch/Data/ewatercycle/experiments/containers/Syn_2310a/staticmaps_1000m_Dischma_feb2024_Synthetic_obs_Dischma_Syn_2310a.nc"
sm_syn = xr.open_dataset(path_static)

for var in sm_post.data_vars:
    diff = sm_post[var] - sm_syn[var]
    print(var,diff.sum().item())
    if diff.sum() != 0:
        print(f"{var} has a difference of {diff.sum()}")
    

folder = "/home/pwiersma/scratch/Data/ewatercycle/experiments/data/output_Synthetic_obs_Dischma_Syn_2310a"
file = xr.open_dataset(join(folder, "output_Synthetic_obs_Dischma_Syn_2310a.nc"))