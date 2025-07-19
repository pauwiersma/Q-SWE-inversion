import xarray as xr
from os.path import join
import numpy as np
# from swissreframe import Coordinate,REFRAME
import matplotlib.pyplot as plt
import glob


MS_path = "/home/pwiersma/scratch/Data/MeteoSwiss"

# Rhires = xr.open_dataset(join(MS_path,"RhiresD_1991_2017/RhiresD_ch01r.swisscors_199101010000_199112310000.nc"))
#
# r= REFRAME()
#
# #result2 = r.compute_gpsref(coordinates = (2600000.1, 1200000.1), transformation='lv95_to_etrf93_geographic')
#
# # coords = map(r.compute_gpsref(transformation = 'lv95_to_etrf93_geographic'),Rhires.chx
#
# lat = Rhires.lat.data.flatten()
# lon = Rhires.lon.data.flatten()
# P  = Rhires.RhiresD.data.flatten()

#%% Convert Rhires to ERA format and save
# Rhires_files = glob.glob(join(MS_path,'RhiresD_1961_2017_ch02_lonlat/*.nc'))
Rhires_files = glob.glob(join(MS_path,'RhiresD_ch01_to_ch02_latlon/*.nc'))

for file in Rhires_files:
    output_file = join(MS_path,'RhiresD_1961_2023_ERA5format','RhiresD_ERA5format'+file[-29:])
    if os.path.isfile(output_file):
        print(output_file," exists already. Break")
        continue
    orig = xr.open_dataset(file)
    orig = orig.rename({'RhiresD':'pr'}).rename(dict(x = 'lon',y = 'lat'))#.squeeze(dim='dummy',drop=True).drop('longitude_latitude')
    orig['time'] = orig.time.data + np.timedelta64(12,'h')

    resolution = 0.020833333
    lon_bnds = np.array([[orig.lon.data-resolution/2],[orig.lon.data+resolution/2]]).squeeze().transpose()
    lat_bnds = np.array([[orig.lat.data-resolution/2],[orig.lat.data+resolution/2]]).squeeze().transpose()
    time_bnds = np.array([[orig.time.data-np.timedelta64(12,'h')],[orig.time.data+np.timedelta64(12,'h')]]).squeeze().transpose()

    lon_bnds_ds = xr.DataArray(lon_bnds,dims=['lon','bnds'],coords={'lon':orig.lon.data}).to_dataset(name='lon_bnds')
    lat_bnds_ds = xr.DataArray(lat_bnds,dims=['lat','bnds'],coords={'lat':orig.lat.data}).to_dataset(name='lat_bnds')
    time_bnds_ds = xr.DataArray(time_bnds,dims=['time','bnds'],coords={'time':orig.time.data}).to_dataset(name='time_bnds')

    R1 = xr.concat([orig,lon_bnds_ds],data_vars='different',dim=['lon','bnds']).drop_dims('concat_dim') # Dimension concat_dim is added automatically, don't know how to prevent that
    R2 = xr.concat([R1,lat_bnds_ds],data_vars='different',dim=['lat','bnds']).drop_dims('concat_dim')
    new = xr.concat([R2, time_bnds_ds], data_vars='different', dim=['time', 'bnds']).drop_dims('concat_dim')
    new.attrs['Conversion']='Converted to ERA5 netcdf format by Pau Wiersma'
    new.to_netcdf(join(MS_path,'RhiresD_1961_2023_ERA5format','RhiresD_ERA5format'+file[-29:]))

#%% Convert Rhires to ERA format and save
# Tabs_files = glob.glob(join(MS_path,'TabsD_1961_2017_ch02_lonlat/*.nc'))
Tabs_files = glob.glob(join(MS_path,'TabsD_ch01_to_ch02_latlon/*.nc'))

for file in Tabs_files:
    orig = xr.open_dataset(file)
    orig = orig.rename({'TabsD':'tas'}).rename(dict(x = 'lon',y = 'lat'))#.squeeze(dim='dummy',drop=True).drop('longitude_latitude')
    orig['time'] = orig.time.data + np.timedelta64(12,'h')

    resolution = 0.020833333
    lon_bnds = np.array([[orig.lon.data-resolution/2],[orig.lon.data+resolution/2]]).squeeze().transpose()
    lat_bnds = np.array([[orig.lat.data-resolution/2],[orig.lat.data+resolution/2]]).squeeze().transpose()
    time_bnds = np.array([[orig.time.data-np.timedelta64(12,'h')],[orig.time.data+np.timedelta64(12,'h')]]).squeeze().transpose()

    lon_bnds_ds = xr.DataArray(lon_bnds,dims=['lon','bnds'],coords={'lon':orig.lon.data}).to_dataset(name='lon_bnds')
    lat_bnds_ds = xr.DataArray(lat_bnds,dims=['lat','bnds'],coords={'lat':orig.lat.data}).to_dataset(name='lat_bnds')
    time_bnds_ds = xr.DataArray(time_bnds,dims=['time','bnds'],coords={'time':orig.time.data}).to_dataset(name='time_bnds')

    R1 = xr.concat([orig,lon_bnds_ds],data_vars='different',dim=['lon','bnds']).drop_dims('concat_dim') # Dimension concat_dim is added automatically, don't know how to prevent that
    R2 = xr.concat([R1,lat_bnds_ds],data_vars='different',dim=['lat','bnds']).drop_dims('concat_dim')
    new = xr.concat([R2, time_bnds_ds], data_vars='different', dim=['time', 'bnds']).drop_dims('concat_dim')
    #Dont forget to convet to K
    new['tas'] = xr.DataArray(new.tas+273.15,attrs=new.tas.attrs)
    new.tas.attrs['units'] = 'K'
    new.attrs['Conversion']='Converted to ERA5 netcdf format by Pau Wiersma'
    new.to_netcdf(join(MS_path,'TabsD_1961_2023_ERA5format','TabsD_ERA5format'+file[-29:]))


#%%
# ERA5 = xr.open_dataset("/home/tesla-k20c/data/pau/ERA5/era5_total_precipitation_1990_hourly.nc")
ERA5 = xr.open_dataset("/home/tesla-k20c/ssd/pau/ewatercycle/ERA5/OBS6/Tier3/ERA5/OBS6_ERA5_reanaly_1_day_tas_19930101-20191231.nc")

#%% Put Pf and T in one file for all years
START_YEAR = 1993
END_YEAR   = 2017

P_like_ERA_all = np.array(glob.glob(join(MS_path,'RhiresD_1961_2017_ERA5format','*.nc')))
P_years = np.array([int(P[-28:-24]) for P in P_like_ERA_all])
P_like_ERA_selection = P_like_ERA_all[(P_years>=START_YEAR)&(P_years<=END_YEAR)]
P_nc = xr.concat([xr.open_dataset(f) for f in P_like_ERA_selection],dim = 'time',data_vars='minimal')
P_nc.to_netcdf(join(MS_path,'MeteoSwiss_like_ERA5_pr_1993_2017.nc'))

T_like_ERA_all = np.array(glob.glob(join(MS_path,'TabsD_1961_2017_ERA5format','*.nc')))
T_years = np.array([int(T[-28:-24]) for T in T_like_ERA_all])
T_like_ERA_selection = T_like_ERA_all[(T_years>=START_YEAR)&(T_years<=END_YEAR)]
T_nc = xr.concat([xr.open_dataset(f) for f in T_like_ERA_selection],dim = 'time',data_vars = 'minimal')
T_nc.to_netcdf(join(MS_path,'MeteoSwiss_like_ERA5_tas_1993_2017.nc'))


nc_final = xr.concat([P_nc,T_nc],dim = ['time','lat','lon','bnds'],data_vars = 'different')

nc_final.to_netcdf(join(MS_path,'MeteoSwiss_like_ERA5_pr_tas_1993_2017.nc'))

# bbox =
# nc_final_cropped =




