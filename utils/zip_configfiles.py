import os
import glob
import shutil
import zipfile


# List of config files
config_files = glob.glob("/home/pwiersma/scratch/Data/ewatercycle/experiments/sbm_config*")

# Define the name of the zip file
zip_filename = "/home/pwiersma/scratch/Data/ewatercycle/experiments/sbm_configs_12_9_24.zip"

# Create a zip file and add each config file to it
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for i,file in enumerate(config_files):
        if not "sbm_config_CH_orig.toml" in file:
            zipf.write(file, os.path.basename(file))
        else:
            print("Ignoring sbm_config_CH_orig.toml")
        if i%1000==0:
            print(i)    

for i,file in enumerate(config_files):
        if not "sbm_config_CH_orig.toml" in file:
            if not 'zip' in file:
                print(file)
            # zipf.write(file, os.path.basename(file))
                os.remove(file)
        else:
            print("Ignoring sbm_config_CH_orig.toml")
        if i%1000==0:
            print(i)    


print(f"All config files have been zipped into {zip_filename}")