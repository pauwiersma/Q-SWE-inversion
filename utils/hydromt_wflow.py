#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 18:14:31 2023

@author: tesla-k20c

Just a file for plotting and visualizing


"""
from hydromt.models.model_grid import GridModel
from hydromt.config import configread
root ="/mnt/scratch_pwiersma/PauWiersma/Data/HydroMT"
import os 
os.chdir(root)


import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os
#%%


ds = xr.open_dataset(os.path.join(root,"flow_dem_test_thur_smallbbox.nc"))
da_flw = xr.open_dataset(os.path.join(root,"da_flw_debug_thur_smallbbox.nc"))['flwdir']


ds.raster.set_crs('epsg:4326')
da_flw.raster.set_crs(ds.raster.crs)

ds_out = da_flw.to_dataset().reset_coords(["x_out", "y_out"])


ds_out.coords["mask"] = ds["mask"].astype(np.int8).raster.reproject_like(da_flw, method="nearest").astype(np.bool)

bas_mask = xr.open_dataset(os.path.join(root,"bas_mask9.nc"))

soilgrids = xr.open_dataset(os.path.join(root,"soilgrids_ds.nc"))


soilgrids_path = "/mnt/scratch_pwiersma/PauWiersma/Data/HydroMT/static_data/base/soilgrids/"
soilthickness = xr.open_rasterio(os.path.join(soilgrids_path,'v2017/SoilThickness_250m_ll.tif')).squeeze()
random2020file =  xr.open_rasterio(os.path.join(soilgrids_path,'v2020/bdod_0-5cm_mean_250m_ll.tif')).squeeze()
soilthickness_thur = soilthickness.rio.clip_box ( 8,47,10,48,crs = "EPSG:4326").interp_like(random2020file)
soilthickness_thur.rio.to_raster(os.path.join(soilgrids_path,"v2020/SoilThickness_250m_ll.tif"))
#%%
# data_libs = [r'/path/to/data_catalog.yml']
# model_root = r'/path/to/model_root
# opt=configread(r'/path/to/grid_model_config.ini')  # parse .ini configuration
# mod = GridModel(model_root, data_libs=data_libs)  # initialize model with default logger
# mod.build(region={'bbox': [4.6891,52.9750,4.9576,53.1994]}, opt=opt)

#%% Batyhmetry and river stuff

gdf_out = gpd.read_file(os.path.join(root,"gdf_out.shp"))
gdf_riv = gpd.read_file(os.path.join(root,"gdf_riv.shp"))

gdf_out_hydroatlas = gpd.read_file(os.path.join(root,"gdf_out_hydroatlas.shp"))
gdf_riv_hydroatlas = gpd.read_file(os.path.join(root,"gdf_riv_hydroatlas.shp"))

#%% staticmaps 
staticmaps_jonschwil = xr.open_dataset(os.path.join(root, "model_builds/Jonschwil_soilgrids2020/staticmaps.nc"))
old_staticmaps = xr.open_dataset("/home/tesla-k20c/ssd/pau/ewatercycle/parameter-sets/wflow_thur_sbm_nc_1000_JS/staticmaps.nc")
staticmaps_hydroATLAS = xr.open_dataset(os.path.join(root, "model_builds/Jonschwil_soilgrids2020_HydroATLAS/staticmaps.nc"))

staticmaps_30m = xr.open_dataset(os.path.join(root,"model_builds/Jonschwil_soilgrids2017_30m/staticmaps.nc"))
#%% Plot staticmaps 

def plot_staticmaps(staticmap):
    for var in staticmap.variables:
        if var in ['layer','c','LAI','spatial_ref','time']:
            continue
        plt.figure()
        staticmap[var].plot()
        plt.show()
        
        
#%% Create staticmaps for multiple catchments
centers = pd.read_csv("/home/pwiersma/scratch/Data/HydroMT/catchment_centers.csv",)
center_list = []
inifile_name = "wflow_aug2023.ini"
subprocess.Popen("conda activate hydromt-wflow",shell=True)
for row in range(len(centers)):
    # center_list.append([centers.Center_lon[row],centers.Center_lat[row]])
    center = [centers.Center_lon[row],centers.Center_lat[row]]
    name = centers.Name[row]
    folder_name = f"/home/pwiersma/scratch/Data/HydroMT/model_builds/wflow_{name}_aug2023"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    command = f"hydromt build wflow {folder_name} -r \"{{'subbasin':{center},'strord':4}}\" -i wflow_CH_aug2023.ini -d data_sources_soilgrids2017_hydroATLASfix.yml -vv"
    print(command)
    result = subprocess.Popen(command,text = True, shell =True)