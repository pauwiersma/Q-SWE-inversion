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
import ewatercycle.forcing
import ewatercycle.models
import xarray as xr
import rioxarray as rioxr
from scipy import interpolate
import matplotlib.pyplot as plt
import time
ewatercycle.__version__

os.chdir("/home/pwiersma/scratch/Data/ewatercycle")
import iris
from esmvaltool.diag_scripts.hydrology.wflow import *
import datetime
import cftime
import cf_units

#%% Settings


centers = pd.read_csv("/home/pwiersma/scratch/Data/ewatercycle/aux_data/gauges/gaugesCH.csv")
centers = centers[centers.use=='y']
basin_names = centers.name.values


ROOTDIR = "/home/pwiersma/scratch/Data/ewatercycle"
OUTDIR = join(ROOTDIR,"wflow_Julia_forcing")
os.chdir(ROOTDIR)

START_YEAR = 2004
END_YEAR = 2015
RESOLUTION = '1000m'
HYDROMT_POSTFIX = 'aug2023'


#%% Concat forcing files for entire CH
variables = ['RhiresD','TabsD']

for var in variables:
    input_folder = f"/home/pwiersma/scratch/Data/MeteoSwiss/{var}_1961_2017_ERA5format/"
    output_file = f"/home/pwiersma/scratch/Data/MeteoSwiss/{var}_{START_YEAR}_{END_YEAR}_CH_ERA5format.nc"
    
    if os.path.isfile(output_file):
        print("CH forcing already exists, continue")
        continue
    
    datasets = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        year_str = str(year)
        file_pattern = f"{var}_ERA5format_{year_str}01010000_{year_str}12310000.nc"
        file_path = os.path.join(input_folder, file_pattern)
        
        if os.path.exists(file_path):
            dataset = xr.open_dataset(file_path)
            datasets.append(dataset)
    
    combined_dataset = xr.concat(datasets, dim="time")
    combined_dataset.to_netcdf(output_file)
    
    
#%% Loop through basins
for basin in basin_names: 
    
    basename = '_'.join([
        'wflow',
        'MeteoSwiss',
        RESOLUTION,
        basin,
        str(START_YEAR),
        str(END_YEAR),
    ])
    output_file =join(OUTDIR,f"{basename}.nc")
    if os.path.isfile(output_file):
        print(output_file," exists already. Continue")
        continue
    
    #%%
    """Process data for use as input to the wflow hydrological model."""
    
    # for dataset, metadata in group_metadata(input_metadata, 'dataset').items():
    all_vars = {}
    all_vars['tas'] = iris.load_cube(f"/home/pwiersma/scratch/Data/MeteoSwiss/TabsD_{START_YEAR}_{END_YEAR}_CH_ERA5format.nc",'tas') 
    all_vars['pr'] = iris.load_cube(f"/home/pwiersma/scratch/Data/MeteoSwiss/RhiresD_{START_YEAR}_{END_YEAR}_CH_ERA5format.nc",'pr') 
    for var in all_vars.values():
        var.coords('latitude')[0].guess_bounds()
        var.coords('longitude')[0].guess_bounds()
        shift_era5_time_coordinate(var)

    
    
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
    # gtopo = iris.load_cube("/home/pwiersma/scratch/Data/MeteoSwiss/DEM/GTOPO_Europe.nc")
    # gtopo = rioxr.open_rasterio("/home/pwiersma/scratch/Data/MeteoSwiss/DEM/gt30w020n90.tif")
    # gtopo = gtopo.rio.clip_box(minx=MS_nc.lon.min(),maxx=MS_nc.lon.max(),miny = MS_nc.lat.min(),maxy=MS_nc.lat.max())
    # gtopo = gtopo.rename({'x':'lon','y':'lat'}).squeeze(drop=True)
    # gtopo = gtopo.interp_like(MS_nc.pr).where(~np.isnan(MS_nc.pr[0])).drop_vars('time')
    # gtopo.to_dataset(name='surface_elevation').to_netcdf("/home/pwiersma/scratch/Data/MeteoSwiss/DEM/GTOPO_Europe.nc")
    gtopo_cube = iris.load_cube("/home/pwiersma/scratch/Data/MeteoSwiss/DEM/GTOPO_Europe.nc")
    all_vars['orog'] = gtopo_cube
    
    # logger.info("Processing variable precipitation_flux")
    scheme = 'area_weighted'
    pr_dem = rechunk_and_regrid(all_vars['pr'], iris_dem, scheme)
    
    logger.info("Processing variable temperature")
    tas_dem = regrid_temperature(
        all_vars['tas'],
        all_vars['orog'],
        iris_dem,
        scheme,
    )
    
    all_vars['psl'] = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_psl_19930101-20191231.nc")[0][:9131] #only until 2017
    all_vars['rsds'] = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_rsds_19930101-20191231.nc")[0][:9131]
    all_vars['rsdt'] = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_CFday_rsdt_19930101-20191231.nc")[0][:9131]
    
    
    time_constraint = iris.Constraint(time=lambda cell: START_YEAR<= cell.point.year <= END_YEAR)
    # test = iris.load_cube("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_pr_19930101-20191231.nc")[0]
    
    
    # logger.info("Processing variable potential evapotranspiration") #apprently this didnt' work 
    # if 'evspsblpot' in all_vars:
    #     pet = all_vars['evspsblpot']
    #     pet_dem = rechunk_and_regrid(pet, dem, scheme)
    # else:
    #     logger.info("Potential evapotransporation not available, deriving")
    psl_dem = rechunk_and_regrid(all_vars['psl'], iris_dem, scheme).extract(time_constraint)
    rsds_dem = rechunk_and_regrid(all_vars['rsds'], iris_dem, scheme).extract(time_constraint)
    rsdt_dem = rechunk_and_regrid(all_vars['rsdt'], iris_dem, scheme).extract(time_constraint)
    
    # print(tas_dem.coord('time').points[0])
    def ref_to_ref(var,ref1,ref2):
        """Take MeteoSwiss_like_ERA5 inputs and convert the time units to days since ref2-1-1"""
        t_coord = var.coord('time')
        t_unit = t_coord.units
        new_t_unit_str = f'days since {ref2}-01-01 00:00:00'
        new_t_unit = cf_units.Unit(new_t_unit_str,calendar = cf_units.CALENDAR_STANDARD)
        ref_date = datetime.datetime(ref1,1,1)
        timedeltas = np.array([datetime.timedelta(d) for d in t_coord.points])
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
    # pet_dem.var_name = 'pet'
    
    
    # from esmvaltool.diag_scripts.hydrology.derive_evspsblpot import *
    # delta_svp = tetens_derivative(tas_dem)
    # gamma, cs_const, beta, lambda_ = get_constants(psl_dem)
    
    # the definition of the radiation components according to the paper:
    # kdown = rsds_dem
    # kdown_ext = rsdt_dem
    # Equation 5
    # rad_factor = np.float32(1 - 0.23)
    # net_radiation = (rad_factor * kdown) - (kdown * cs_const / kdown_ext)
    # Equation 6
    # the unit is W m-2
    # tas_unit = tas_dem.coords('time')[0].units
    # psl_unit = psl_dem.coords('time')[0].units
    # l = [delta_svp,gamma]
    # l2 = iris.util.equalise_attributes(l)
    
    # ref_evap = ((delta_svp.data / (delta_svp.data + gamma.data)) * net_radiation.data) + beta.points
    
    # pet = ref_evap / lambda_
    # pet.var_name = 'evspsblpot'
    # pet.standard_name = 'water_potential_evaporation_flux'
    # pet.long_name = 'Potential Evapotranspiration'
    
    
    
    
    
    
    
    # logger.info("Converting units")
    pet_dem.units = pet_dem.units / 'kg m-3'
    pet_dem.data = pet_dem.core_data() / 1000
    pet_dem.convert_units('mm day-1')
    
    pet_dem.rename('pet')
    
    # pr_dem.units = pr_dem.units / 'kg m-3'
    # pr_dem.data = pr_dem.core_data() / 1000.
    # pr_dem.convert_units('mm day-1')
    
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








#%% IRIS test

# # ERA5_nc = xr.open_dataset("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_pr_19930101-20191231.nc")
# # # ERA5_cube = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_pr_19930101-20191231.nc")
# ERA5_cube = iris.load_cube("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_tas_19930101-20191231.nc")
# # # You can use a * wildcard to load all nc files matching that pattern
# # # and then specify to load only the cubes with varname x
# # # ERA5 = iris.load("/home/pwiersma/scratch/Data/ewatercycle/ERA5/OBS6/Tier3/ERA5/*",'air_temperature')

# # # MS = iris.load("/home/pwiersma/scratch/Data/MeteoSwiss/MeteoSwiss_like_ERA5_pr_1993_2017.nc")
# MS_nc = xr.open_dataset("/home/pwiersma/scratch/Data/MeteoSwiss/MeteoSwiss_like_ERA5_pr_1993_2017.nc")
# # MS_nc = xr.open_dataset("/home/pwiersma/scratch/Data/MeteoSwiss/RhydchprobD/RhydchprobD_latlon_linear_200m/RhydchprobD_latlon_linear_200m_1991_2010_member1.nc")


# # # MS = iris.load("/home/pwiersma/scratch/Data/MeteoSwiss/MeteoSwiss_like_ERA5_pr_1993_2017_CF17.nc")
# # MS = iris.load_cube("/home/pwiersma/scratch/Data/MeteoSwiss/MeteoSwiss_like_ERA5_pr_1993_2017.nc",'daily precipitation sum')
# # MS.coord('longitude').guess_bounds()
# # MS.coord('latitude').guess_bounds()


# # MS_regrid = MS.regrid(ERA5_cube,iris.analysis.AreaWeighted())



# ERA5 = iris.load("/home/pwiersma/scratch/Data/ewatercycle/esmvaltool_output/recipe_wflow_Jonschwil_200m_1993_2005/work/wflow_daily/script/wflow_ERA5_Thur_Jonschwil_1993_2005.nc")
# ERA5_nc = xr.open_dataset("/home/pwiersma/scratch/Data/ewatercycle/esmvaltool_output/recipe_wflow_Jonschwil_200m_1993_2005/work/wflow_daily/script/wflow_ERA5_Thur_Jonschwil_1993_2005.nc")


    
    # Old Wflow reading
    # Interpolating variables onto the dem grid
    # Read the target cube, which contains target grid and target elevation
    # dem_path = Path(cfg['auxiliary_data_dir']) / cfg['dem_file']
    # dem_path = Path("/home/pwiersma/scratch/Data/ewatercycle/parameter-sets/wflow_julia_1000_thurJS/staticmaps/wflow_dem.map")
    # dem = load_dem(dem_path)
    # check_dem(dem, all_vars['pr'])