#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:03:21 2023

@author: pwiersma
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:20:42 2022

@author: tesla-k20c
"""

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
import rioxarray as rioxr
from scipy import interpolate
import matplotlib.pyplot as plt
import time
# ewatercycle.__version__

os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')
import iris
from esmvaltool.diag_scripts.hydrology.wflow import *
import datetime
import cftime
import cf_units

#%% Settings


# def generate_MS_forcing(fname, basin,resolution, start_year,end_year):
#     long_forcing = xr.open_dataset(f"/home/pwiersma/scratch/Data/ewatercycle/wflow_Julia_forcing/wflow_MeteoSwiss_1000m_{basin}_2003_2015.nc")
#     short_forcing = long_forcing.sel(time = slice(str(start_year),str(end_year)))
#     output_file =fname
#     if os.path.isfile(output_file):
#         print(output_file," exists already. Break")
#         return
#     else:
#         print(f"Generating forcing for {basin} between {start_year} and {end_year}")
#     short_forcing.to_netcdf(fname)
#     return 

def generate_MS_forcing(fname, 
                        basin,
                        resolution,
                        start_year,
                        end_year):
    
    ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')
    OUTDIR = join(ROOTDIR,"wflow_Julia_forcing")
    # os.chdir(ROOTDIR)
    
    START_YEAR = start_year
    END_YEAR = end_year
    RESOLUTION = resolution
    HYDROMT_POSTFIX = 'feb2024' #'aug2023'   #
    
    # Concat forcing files for entire CH
    variables = ['RhiresD','TabsD']
    
    for var in variables:
        input_folder = f"/home/pwiersma/scratch/Data/MeteoSwiss/{var}_1961_2023_ERA5format/"
        output_file = f"/home/pwiersma/scratch/Data/MeteoSwiss/{var}_{START_YEAR}_{END_YEAR}_CH_ERA5format.nc"
        
        if os.path.isfile(output_file):
            print("CH forcing already exists, continue")
            continue
        else:
            print(f"Generating {var} for whole CH between {START_YEAR} and {END_YEAR}")
        
        datasets = []
        
        for year in range(START_YEAR, END_YEAR + 1):
            year_str = str(year)
            # file_pattern = f"{var}_ERA5format_{year_str}01010000_{year_str}12310000.nc"
            file_pattern = f"{var}_ERA5format_to_ch02_{year_str}0101_{year_str}1231.nc"

            file_path = os.path.join(input_folder, file_pattern)
            
            if os.path.exists(file_path):
                dataset = xr.open_dataset(file_path)
                datasets.append(dataset)
            else:
                print(f"Year {year} not found, break")
        
        combined_dataset = xr.concat(datasets, dim="time")
        combined_dataset.to_netcdf(output_file)
        
    #Now clip for specific basin according to the mask from the hydromt staticmaps.nc file
             
    basename = '_'.join([
        'wflow',
        'MeteoSwiss',
        RESOLUTION,
        basin,
        str(START_YEAR),
        str(END_YEAR),
    ])
    output_file =fname
    if os.path.isfile(output_file):
        print(output_file," exists already. Break")
        return
    else:
        print(f"Generating forcing for {basin} between {START_YEAR} and {END_YEAR}")
    #%%
    """Process data for use as input to the wflow hydrological model."""
    
    # for dataset, metadata in group_metadata(input_metadata, 'dataset').items():
    all_vars = {}
    all_vars['tas'] = iris.load_cube(f"/home/pwiersma/scratch/Data/MeteoSwiss/TabsD_{START_YEAR}_{END_YEAR}_CH_ERA5format.nc",'tas') 
    all_vars['pr'] = iris.load_cube(f"/home/pwiersma/scratch/Data/MeteoSwiss/RhiresD_{START_YEAR}_{END_YEAR}_CH_ERA5format.nc",'pr') 
    for var in all_vars.values():
        var.coords('latitude')[0].guess_bounds()
        var.coords('longitude')[0].guess_bounds()
        # shift_era5_time_coordinate(var)
    
    
    
    #%% Julia alternative 
    HYDROMT_DIR ="/home/pwiersma/scratch/Data/HydroMT/model_builds"
    staticmaps_hydroATLAS = xr.open_dataset(os.path.join(HYDROMT_DIR, f"wflow_{basin}_{RESOLUTION}_{HYDROMT_POSTFIX}/staticmaps.nc"))
    xr_dem = staticmaps_hydroATLAS.wflow_dem
    iris_dem = xr_dem.to_iris()
    iris_dem.rename('height')
    iris_dem.units = 'm'
    iris_dem.remove_coord('spatial_ref')
    iris_dem.coord('latitude').guess_bounds()
    iris_dem.coord('longitude').guess_bounds()    
    #%%
    
    ### Read GTOPO DEM whioch is used by meteoswiss
    gtopo_cube = iris.load_cube("/home/pwiersma/scratch/Data/MeteoSwiss/DEM/GTOPO_Europe.nc")
    all_vars['orog'] = gtopo_cube
    
    # logger.info("Processing variable precipitation_flux")
    scheme = 'area_weighted'
    pr_dem = regrid(all_vars['pr'], iris_dem, scheme)
    
    logger.info("Processing variable temperature")
    tas_dem = regrid_temperature(
        all_vars['tas'],
        all_vars['orog'],
        iris_dem,
        scheme,
    )
    # test = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6_1993_2019/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_psl_19930101-20191231.nc")[0][:9131] #only until 2017

    # all_vars['psl'] = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_psl_19930101-20191231.nc")[0][:9131] #only until 2017
    # all_vars['rsds'] = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_rsds_19930101-20191231.nc")[0][:9131]
    # all_vars['rsdt'] = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_CFday_rsdt_19930101-20191231.nc")[0][:9131]
    
    all_vars['psl'] = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_psl_19930101-20231231.nc")[0]#only until 2017
    all_vars['rsds'] = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_rsds_19930101-20231231.nc")[0]
    all_vars['rsdt'] = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_CFday_rsdt_19930101-20231231.nc")[0]
    
    time_constraint = iris.Constraint(time=lambda cell: START_YEAR<= cell.point.year <= END_YEAR)
    psl_dem = regrid(all_vars['psl'], iris_dem, scheme).extract(time_constraint)
    rsds_dem = regrid(all_vars['rsds'], iris_dem, scheme).extract(time_constraint)
    rsdt_dem = regrid(all_vars['rsdt'], iris_dem, scheme).extract(time_constraint)
    
    # print(tas_dem.coord('time').points[0])
    def ref_to_ref(var,ref1,ref2):
        """Take MeteoSwiss_like_ERA5 inputs and convert the time units to days since ref2-1-1"""
        t_coord = var.coord('time')
        t_unit = t_coord.units
        new_t_unit_str = f'days since {ref2}-01-01 00:00:00'
        new_t_unit = cf_units.Unit(new_t_unit_str,calendar = cf_units.CALENDAR_STANDARD)
        ref_date = datetime.datetime(ref1,1,1)
        timedeltas = np.array([datetime.timedelta(int(d)) for d in t_coord.points])
        # dtfrom2001 = ref_date+timedeltas
        new_ref = datetime.datetime(ref2,1,1)
        ref_dif = ref_date - new_ref
        new_dt = ref_dif + timedeltas - datetime.timedelta(seconds = 1800) + datetime.timedelta(days = 0.47916667)
        new_dt_data = [dt.days  for dt in new_dt] #(41400/(3600*24)) + (288/(3600*1000000) + 0.47916667
        new_t_coord = iris.coords.DimCoord(new_dt_data,standard_name = 'time',units = new_t_unit)
        new_t_coord.guess_bounds()
        t_coord_dim = var.coord_dims('time')
        var.remove_coord('time')
        var.add_dim_coord(new_t_coord, t_coord_dim)
        return
    ref_to_ref(tas_dem,START_YEAR,1850)
    ref_to_ref(pr_dem,START_YEAR,1850)
    # print(tas_dem.coord('time').points[0])
    
    
    def date_to_int(var):
        """Take ERA5 inputs and make sure the timestamps are integers"""
        coord = var.coord('time')
        unit = coord.units
        points = coord.points
        new_points = [int(p) for p in points]
        new_coord = iris.coords.DimCoord(new_points,standard_name = 'time',units = unit)
        coord_dim = var.coord_dims('time')
        var.remove_coord('time')
        var.add_dim_coord(new_coord, coord_dim)
        return
    for v in [psl_dem,rsds_dem,rsdt_dem]:
        date_to_int(v)
        
    
    pet_dem = debruin_pet(
        tas=tas_dem,
        psl=psl_dem,
        rsds=rsds_dem,
        rsdt=rsdt_dem,
    )
    
    
    # logger.info("Converting units")
    pet_dem.units = pet_dem.units / 'kg m-3'
    pet_dem.data = pet_dem.core_data() / 1000
    pet_dem.convert_units('mm day-1')
    
    pet_dem.rename('pet')
    tas_dem.convert_units('degC')
    
    # Adjust longitude coordinate to wflow convention
    for cube in [tas_dem, pr_dem]:#, pet_dem]:
        cube.coord('longitude').points = (cube.coord('longitude').points +
                                          180) % 360 - 180
    
    cubes = iris.cube.CubeList([pr_dem, tas_dem, pet_dem])
    
    
    # time_coord = cubes[0].coord('time')
    # START_YEAR = time_coord.cell(0).point.year
    # END_YEAR = time_coord.cell(-1).point.year
    
    # logger.info("Saving cubes to file %s", output_file)
    iris.save(cubes, output_file, fill_value=1.e20)
    # MS_new_nc = xr.open_dataset(output_file)
    # 







