# Import necessary libraries
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os

# Change the current working directory to the specified path
os.chdir("/home/pwiersma/scratch/Data/HydroMT")

# Load data from a CSV file into a Pandas DataFrame
centers = pd.read_csv("/home/pwiersma/scratch/Data/ewatercycle/aux_data/gauges/gaugesCH.csv")

# Create an empty list to store center coordinates
center_list = []
print("Everything is fine")

# Set the name of the configuration file
inifile_name = "wflow_CH_feb2024.ini"

# Loop through each row in the 'centers' DataFrame
for row in range(len(centers)):
    if centers.use[row] =='n':
        continue
    # Extract center coordinates and name from the DataFrame
    center = [centers.lon[row], centers.lat[row]]
    name = centers.name[row]
    print("Name: ", name)
    if name != 'Riale_di_Calneggia':
        continue
    
    # Create a folder name based on the center's name
    folder_name = f"/home/pwiersma/scratch/Data/HydroMT/model_builds/wflow_{name}_50m_feb2024"
    
    # Check if the folder already exists; if not, create it
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    
    # Construct the command for building the hydrological model using hydromt
    command = (
        f"hydromt build wflow {folder_name} -r \"{{'subbasin':{center},'strord':4}}\" "
        f"-i {inifile_name} -d data_sources_soilgrids2017_hydroATLASfix.yml -vv"
    )
    
    # Print the command for debugging purposes
    print(command)
    
    # Execute the command using subprocess with text output
    result = subprocess.Popen(command, text=True, shell=True)
