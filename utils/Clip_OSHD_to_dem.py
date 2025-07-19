# -*- coding: utf-8 -*-

import xarray as xr

import rioxarray as rioxr

import numpy as np
from rasterio.enums import Resampling

import matplotlib.pyplot as plt

from rasterio.enums import Resampling
import xarray as xr
import os
import matplotlib.pyplot as plt
from shapely.geometry import box
import geopandas as gpd
import os

#%% 1km
# OSHD = xr.open_dataset(f"/home/pwiersma/scratch/Data/SLF/TI_1998_2022_OSHD_1km.nc",decode_times = False).rio.write_crs('epsg:21781')
OSHD = xr.open_dataset(f"/home/pwiersma/scratch/Data/SLF/OSHD/OSHD_1km_bilinear.nc").rio.write_crs('epsg:4326')


# OSHD = OSHD.rename(dict(xx = 'x', yy = 'y'))
resolution = '250m'

# for basin in ['Mogelsberg','Dischma','Riale_di_Pincascia','Ova_da_Cluozza','Sitter','Kleine_Emme','Werthenstein',
#           'Sense','Ilfis','Eggiwil','Chamuerabach','Veveyse','Minster','Ova_dal_Fuorn','Alp','Biber','Muota','Riale_di_Calneggia',
#           'Chli_Schliere','Allenbach','Jonschwil','Landwasser','Landquart','Rom','Verzasca']:
# # for basin in ['Riale_di_Calneggia']:
#     dem_path = f"/home/pwiersma/scratch/Data/ewatercycle/wflow_staticmaps/staticmaps_{resolution}_{basin}_feb2024.nc"
#     if not os.path.isfile(dem_path):
#         print(basin, " no dem")
#         continue
#     else: 
#         print("Resampling ", basin)
#         dem = xr.open_dataset(dem_path)['wflow_dem'].rename(dict(lon = 'x',lat = 'y')).rio.write_crs('epsg:4326')
#         # plt.figure()
#         # dem.plot()
#         # # plt.xlim(8.45,8.55)
#         # # plt.ylim(46.32,46.4)
#         # plt.title(basin)
#         # plt.show()

#         # Clip OSHD to the extent of the dem
#         geometry = gpd.GeoDataFrame(index=[0], crs=dem.rio.crs, geometry=[box(*dem.rio.bounds())])
        
#         OSHD_clipped = OSHD.rio.clip(geometry.geometry.buffer(0.05))

#         new_OSHD_path = f"/home/pwiersma/scratch/Data/SLF/OSHD/OSHD_{resolution}_latlon_{basin}.nc"
#         new_OSHD = OSHD_clipped.rio.reproject_match(dem, resampling=Resampling.bilinear)
#         # new_OSHD = OSHD.rio.reproject_match(dem, resampling=Resampling.bilinear)

#         del new_OSHD.romc_all.attrs['grid_mapping']
#         del new_OSHD.dem.attrs['grid_mapping']
#         del new_OSHD.swee_all.attrs['grid_mapping']
#         del new_OSHD.forest.attrs['grid_mapping']
#         del new_OSHD.open.attrs['grid_mapping']
#         # new_OSHD['swee_all'].plot()
#         # # plt.xlim(8.45,8.55)
#         # # plt.ylim(46.32,46.4)
#         # plt.show()
        
#         # new_OSHD['dem'].plot()
#         # # plt.xlim(8.45,8.55)
#         # # plt.ylim(46.32,46.4)
#         # plt.show()
        
#         # dem_diff = new_OSHD['dem'] - dem
#         # dem_diff.plot(vmin = -500,vmax = 500,cmap = "RdBu")
#         # plt.show()
        
#         new_OSHD.to_netcdf(new_OSHD_path)
        
#%% 250m
years = np.arange(2015,2022)
FSM = dict()
for year in years:
    FSM[year] = xr.open_dataset(f"/home/pwiersma/scratch/Data/SLF/FSM/FSM_250m_bilinear_{year}.nc",
                                chunks = 'auto').rio.write_crs('epsg:4326')

# FSM = FSM.rename(dict(xx = 'x', yy = 'y'))
resolution = '50m'

# for basin in ['Mogelsberg','Dischma','Riale_di_Pincascia','Ova_da_Cluozza','Sitter','Kleine_Emme','Werthenstein',
#           'Sense','Ilfis','Eggiwil','Chamuerabach','Veveyse','Minster','Ova_dal_Fuorn','Alp','Biber','Muota','Riale_di_Calneggia',
#           'Chli_Schliere','Allenbach','Jonschwil','Landwasser','Landquart','Rom','Verzasca']:
for basin in ['Riale_di_Calneggia']:
    dem_path = f"/home/pwiersma/scratch/Data/ewatercycle/wflow_staticmaps/staticmaps_{resolution}_{basin}_feb2024.nc"
    if not os.path.isfile(dem_path):
        print(basin, " no dem")
        continue
    else: 
        print("Resampling ", basin)
        dem = xr.open_dataset(dem_path)['wflow_dem'].rename(dict(lon = 'x',lat = 'y')).rio.write_crs('epsg:4326')
        # plt.figure()
        # dem.plot()
        # # plt.xlim(8.45,8.55)
        # # plt.ylim(46.32,46.4)
        # plt.title(basin)
        # plt.show()

        # Clip FSM to the extent of the dem
        geometry = gpd.GeoDataFrame(index=[0], crs=dem.rio.crs, geometry=[box(*dem.rio.bounds())])
        
        FSM_clipped = dict()
        for year in years:
            FSM_clipped[year] = FSM[year].rio.clip(geometry.geometry)
            
        FSM_concat = xr.concat([FSM_clipped[y] for y in years], dim = 'time', data_vars = 'minimal') 

        new_FSM_path = f"/home/pwiersma/scratch/Data/SLF/FSM_{resolution}_latlon_{basin}.nc"
        new_FSM = FSM_concat.rio.reproject_match(dem, resampling=Resampling.bilinear)
        # new_FSM = FSM.rio.reproject_match(dem, resampling=Resampling.bilinear)

        del new_FSM.romc_all.attrs['grid_mapping']
        del new_FSM.dem.attrs['grid_mapping']
        del new_FSM.swet_all.attrs['grid_mapping']
        del new_FSM.forest.attrs['grid_mapping']
        # del new_FSM.open.attrs['grid_mapping']
        # new_FSM['swee_all'].plot()
        # # plt.xlim(8.45,8.55)
        # # plt.ylim(46.32,46.4)
        # plt.show()
        
        # new_FSM['dem'].plot()
        # # plt.xlim(8.45,8.55)
        # # plt.ylim(46.32,46.4)
        # plt.show()
        
        # dem_diff = new_FSM['dem'] - dem
        # dem_diff.plot(vmin = -500,vmax = 500,cmap = "RdBu")
        # plt.show()
        
        new_FSM.to_netcdf(new_FSM_path)
#%%
plt.figure()
snow = xr.where(np.isnan(dem),np.nan,new_FSM.swet_all.isel(time = 780))
new_FSM.dem.plot(cmap = 'Greys')
snow.plot(vmin =0,cmap = 'Blues')

plt.axis('off')
plt.colorbar(None)