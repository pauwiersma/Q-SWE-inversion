
       
import os
from os.path import join
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.colors import LightSource
from SwissStations import *
from SnowClass import *
import rioxarray as rioxr
import numpy as np
import shapely


def plot_dem_with_hillshade(dem, ax,dem_cmap = 'copper'):

    data = dem.values

    # Get the extent of the data
    min_lon = dem.lon.min().item()
    max_lon = dem.lon.max().item()
    min_lat = dem.lat.min().item()
    max_lat = dem.lat.max().item()

    # Generate hillshade
    ls = LightSource(azdeg=180, altdeg=45)
    hillshade = ls.hillshade(data, vert_exag=1, dx=1, dy=1)

    # Print metadata
    # print(out_meta)

    # Plot the data using imshow
    img = ax.imshow(data, cmap=dem_cmap, alpha=0.9, extent=(
        min_lon, max_lon, min_lat, max_lat), vmin =1000,vmax = 3000)
    ax.imshow(hillshade, cmap='gray', alpha=0.25, extent=(
        min_lon, max_lon, min_lat, max_lat))
    cbar = plt.colorbar(img, ax=ax, label='Elevation',)
    ax.set_title('Clipped DEM Data with Hillshade')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # plt.colorbar(img, ax=ax, label='Elevation')
    return ax
def create_dhm25_dem(BASIN,output_path,rootdir,RESOLUTION):
    # if not hasattr(self,'dem'):

    ROOTDIR = rootdir
    RESOLUTION = RESOLUTION

    dem =  xr.open_dataset(join(ROOTDIR,'wflow_staticmaps',f'staticmaps_{RESOLUTION}_{BASIN}_feb2024.nc'))['wflow_dem']

    buffer = 0.1
    min_lon = dem.lon.min().item()-buffer
    max_lon = dem.lon.max().item()+buffer
    min_lat = dem.lat.min().item()-buffer
    max_lat = dem.lat.max().item()+buffer
    
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
        
    load_dhm25_dem(rootdir,BASIN,RESOLUTION)

        
def load_dhm25_dem(ROOTDIR,BASIN,RESOLUTION):
        
    dhm25_path = join(ROOTDIR,"aux_data","dhm25_clipped",f"{BASIN}_dhm25_buffed.tif")
    dhm25_path = dhm25_path
    if os.path.exists(dhm25_path):
        dhm25 = rioxr.open_rasterio(dhm25_path).sel(band = 1).squeeze()
        #remove "band" and "spatial_ref" coordinates
        dhm25 = dhm25.drop_vars(['band','spatial_ref'])
        dhm25 = dhm25.rename({'x':'lon','y':'lat'})
        return dhm25 
    else:
        create_dhm25_dem(BASIN,dhm25_path,ROOTDIR,RESOLUTION)


        

def plot_delin_on_ax(BASIN, ax):
    folder = "/home/pwiersma/scratch/Data/Basins"
    directories = os.listdir(folder)
    for directory in directories:
        if BASIN.title()[:5] in directory:
            if not 'zip' in directory:
                number = directory[3:7]
                file  = gpd.read_file(join(folder,directory,f"CH-{number}.shp"))
    delin = file
    file.plot(ax = ax, color = 'none', edgecolor = 'black', linewidth = 0.5)
    S = SwissStation(BASIN)
    ax.plot(S.lon,S.lat, 'ro',label = 'Discharge station')
    return delin

def plot_rivers(BASIN,ax,delin):
    clipped_dir = "/home/pwiersma/scratch/Data/ewatercycle/aux_data/clipped_rivers"
    clipped_file = join(clipped_dir,f"{BASIN}_rivers.shp")
    if os.path.exists(clipped_file):
        rivers_dem = gpd.read_file(clipped_file)
    else:
        rivers_folder = "/home/pwiersma/scratch/Data/GIS/Hydrography/FlussordnungszahlStrahler"
        # rivers_folder = "/home/pwiersma/scratch/Data/HydroMT"
        rivers_file = join(rivers_folder,"FLOZ_epsg4326.shp")
        # rivers_file = join(rivers_folder,"gdf_riv_hydroatlas.shp")

        rivers = gpd.read_file(rivers_file)

        rivers_dem = gpd.clip(rivers, delin.geometry)
        rivers_dem.to_file(clipped_file)
    # Plot the rivers
    rivers_dem.plot(ax = ax, color='blue', linewidth=1, alpha = 0.5)


# # def plot_fancydem(dem,BASIN):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     plot_dem_with_hillshade(dem, ax,dem_cmap = 'copper')
#     delin = plot_delin_on_ax(BASIN, ax)
#     plot_rivers(BASIN,ax,delin)
#     # plot_SWE_as_ax(ax)
#     plt.show()


#Dischma
BASIN = 'Riale_di_Calneggia'
dem = load_dhm25_dem("/home/pwiersma/scratch/Data/ewatercycle",BASIN,'1000m')
dem = xr.where(dem ==0, np.nan, dem)
# plot_fancydem(dem,BASIN)
C = SnowClass(BASIN)
snowstats = dict()
for station in C.station_short_names:
    snowstats[station] = C.station_coordinates[station]

FIGDIR = "/home/pwiersma/scratch/Figures/ewc_figures/DEMs"

fig, ax = plt.subplots(figsize=(6, 6))
plot_dem_with_hillshade(dem, ax,dem_cmap = 'copper')
delin = plot_delin_on_ax(BASIN, ax)
plot_rivers(BASIN,ax,delin)

markers = ['o','s','^','v','<','>','D','P','X','*']
for i,station in enumerate(snowstats.keys()):
    ax.plot(snowstats[station][0],snowstats[station][1],color = 'black',
             marker =markers[i] ,label = station,linestyle = 'None')
if BASIN == 'Riale_di_Calneggia':
    ax.set_xlim(8.45,8.57)
    ax.set_ylim(46.3,46.45)
elif BASIN =='Dischma':
    ax.set_xlim(9.78,9.98)
    ax.set_ylim(46.66,46.85)

ax.legend()
plt.title(BASIN)
# plot_SWE_as_ax(ax)
plt.show()
fig.savefig(join(FIGDIR,f'{BASIN}_39.png'),dpi = 300, bbox_inches = 'tight')
