import logging
import warnings
import numpy as np
import pandas as pd
import glob
import os
from os.path import join
import rasterio as rio
import tomlkit
import shutil

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("esmvalcore")
logger.setLevel(logging.WARNING)
import xarray as xr
import rioxarray as rioxr
from scipy import interpolate
import matplotlib.pyplot as plt 
import time
from datetime import date
import HydroErr as he
import json
import time

EWC_RUNDIR = os.getenv('EWC_RUNDIR', default = '/home/pwiersma/scratch/Scripts/Github/ewc/')
os.chdir(EWC_RUNDIR)
from SwissStations import *
from SnowClass import *
from RunWflow_Julia import *

from wflow_spot_setup import *

from pathlib import Path
import numpy as np
from IPython.display import clear_output

def  Postruns(config, 
             calibration_purpose,
             single_year = None,
             soilmember = None,
             UB_dir = None):
    cfg = config
    
    if calibration_purpose == 'Soilcalib':
        dists = ['prior','posterior']
        k = cfg['SOIL_K']
        start_year = cfg['START_YEAR']-1
        end_year = cfg['END_YEAR']
        model_dics = {}
        for dist in dists:
            FOLDER = join(cfg['OUTDIR'],f"{calibration_purpose}")
            cluster_parsets_path = join(FOLDER, 
                            f"{cfg['BASIN']}_ksoil{k}_{dist}.csv")
            cluster_parsets = pd.read_csv(cluster_parsets_path,header=0, index_col=0)

            cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')

            
            prefix = f"{calibration_purpose}_{cfg['EXP_ID']}_{cfg['BASIN']}_{dist}"

            for member in range(len(cluster_parsets_dic)):
                model_dics[f"{prefix}_ms{member}"] = {**cluster_parsets_dic[member]}
                for par in cfg['param_fixed']:
                    if not par in list(cluster_parsets_dic[member].keys()):
                        print("Setting fixed par ",par)
                        model_dics[f"{prefix}_ms{member}"][par] = cfg['param_fixed'][par]
            # for member, parset in cluster_parsets_dic.items():
            #     model_dics[f"{prefix}_m{member}"] = {**parset, **cfg['param_fixed']}
                print(model_dics[f"{prefix}_ms{member}"])
                

    elif calibration_purpose == 'Yearlycalib':
        dists = ['prior','posterior']
        k = cfg['YEARLY_K']
        start_year = single_year-1
        end_year = single_year
        model_dics = {}
        for dist in dists: 

            soilcluster_parsets_path = join(cfg['SOILDIR'], 
                            f"{cfg['BASIN']}_ksoil{cfg['SOIL_K']}_{'posterior'}.csv")
            soilcluster_parsets = pd.read_csv(soilcluster_parsets_path,header=0, index_col=0).loc[soilmember]
            # soilcluster_parsets_dic = soilcluster_parsets.to_dict(orient = 'index')



            FOLDER = join(cfg['OUTDIR'],f"{calibration_purpose}",f"{single_year}",f"{soilmember}")
            cluster_parsets_path = join(FOLDER, 
                            f"{cfg['BASIN']}_ms{soilmember}_kyearly{k}_{dist}.csv")
            cluster_parsets = pd.read_csv(cluster_parsets_path,header=0, index_col=0)
            cols = cluster_parsets.columns
            for soilpar in soilcluster_parsets.index:
                if not soilpar in cols:
                    cluster_parsets[soilpar] = soilcluster_parsets[soilpar]

            cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')

            prefix = f"{calibration_purpose}_{cfg['EXP_ID']}_{cfg['BASIN']}_{single_year}_{dist}_ms{soilmember}"

            for member in range(len(cluster_parsets_dic)):
                model_dics[f"{prefix}_my{member}"] = {**cluster_parsets_dic[member]}
                for par in cfg['param_fixed']:
                    if not par in list(cluster_parsets_dic[member].keys()):
                        print("Setting fixed par ",par)
                        model_dics[f"{prefix}_my{member}"][par] = cfg['param_fixed'][par]

                print(model_dics[f"{prefix}_my{member}"])
    
    elif calibration_purpose =='OB_calib':
        k = cfg['SOIL_K']
        start_year = single_year-1
        end_year = single_year
        model_dics = {}
        #load soilcalib parsets and add rfcf to each one for this specific year 
        FOLDER = join(cfg['OUTDIR'],f"OSHD",f"{single_year}")
        cluster_parsets_path = join(cfg['SOILDIR'], 
                        f"{cfg['BASIN']}_ksoil{k}_{'posterior'}.csv")
        cluster_parsets = pd.read_csv(cluster_parsets_path,header=0, index_col=0)
        cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')

        prefix = f"{calibration_purpose}_{cfg['EXP_ID']}_{cfg['BASIN']}_{single_year}"

        for member in range(len(cluster_parsets_dic)):
            model_dics[f"{prefix}_ms{member}"] = {**cluster_parsets_dic[member]}
            for par in cfg['param_fixed']:
                if not par in list(cluster_parsets_dic[member].keys()):
                    print("Setting fixed par ",par)
                    model_dics[f"{prefix}_ms{member}"][par] = cfg['param_fixed'][par]
            
            model_dics[f"{prefix}_ms{member}"]['DD'] = 'static'
            model_dics[f"{prefix}_ms{member}"]['sfcf'] = 0
            model_dics[f"{prefix}_ms{member}"]['masswasting'] = False
            model_dics[f"{prefix}_ms{member}"]['petcf_seasonal'] = True
            model_dics[f"{prefix}_ms{member}"][cfg['OSHD']] = True

            #add rfcf 
            rfcf_dir = join(cfg['OUTDIR'],'OSHD',f"{single_year}",f"{member}")
            rfcf_file = join(rfcf_dir,f"{cfg['BASIN']}_{single_year}_ms{member}_{calibration_purpose}.csv")
            rfcf = pd.read_csv(rfcf_file,header=0)
            #find the value in parrfcf column that correponds to the highest value in the like1 column
            rfcf_value = rfcf.loc[rfcf['like1'].idxmax()]['parrfcf']
            model_dics[f"{prefix}_ms{member}"]['rfcf'] = rfcf_value

            print(model_dics[f"{prefix}_ms{member}"])

    elif calibration_purpose == 'UB':
        #Upper Benchmark runs 
        #Load UB parameter sets 
        UB_dir = UB_dir 
        SWE_chosen = '_'.join(os.path.basename(UB_dir).split('_')[1:])
        UB_pars_file = glob.glob(join(UB_dir,f"*_{single_year}.csv"))[0]
        UB_pars = pd.read_csv(UB_pars_file,header=0,index_col=0)

        k = cfg['SOIL_K']
        start_year = single_year-1
        end_year = single_year
        FOLDER = join(UB_dir,f"{single_year}")
        Path.mkdir(Path(FOLDER),exist_ok=True)

        model_dics = {}
        prefix = f"{calibration_purpose}_{cfg['EXP_ID']}_{cfg['BASIN']}_{single_year}_{SWE_chosen}"
        for member in range(len(UB_pars)):
            UB_id = UB_pars.index[member]
            model_dics[f"{prefix}_{UB_id}"] = UB_pars.iloc[member].to_dict()
            for par in cfg['param_fixed']:
                if not par in list(UB_pars.iloc[member].keys()):
                    print("Setting fixed par ",par)
                    model_dics[f"{prefix}_{UB_id}"][par] = cfg['param_fixed'][par]
            if cfg['PERFECT_SOIL']==True:
                for par in cfg['param_kinds'].keys():
                    if cfg['param_kinds'][par] == 'soil':  # Update from self.param_kinds to cfg['param_kinds']
                        print("Setting soil par ",par)
                        model_dics[f"{prefix}_{UB_id}"][par] = cfg['trueparams'][par]
            # model_dics[f"{prefix}_{UB_id}"]['DD'] = 'Hock'

            print(model_dics[f"{prefix}_{UB_id}"])




    else:
        k = cfg['SOIL_K']
        dist = 'posterior'
        start_year = cfg['START_YEAR']-1
        end_year = cfg['END_YEAR']
        FOLDER = join(cfg['OUTDIR'],f"{calibration_purpose}")
        cluster_parsets_path = join(cfg['OUTDIR'],"Soilcalib", 
                            f"{cfg['BASIN']}_ksoil{k}_{dist}.csv")
        cluster_parsets = pd.read_csv(cluster_parsets_path,header=0, index_col=0)

        cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')

        model_dics = {}
        if calibration_purpose == 'OSHD':

            prefix = f"{calibration_purpose}_{cfg['EXP_ID']}_{cfg['BASIN']}_{cfg['OSHD']}"
            for member in range(len(cluster_parsets_dic)):
                model_dics[f"{prefix}_ms{member}"] = {**cluster_parsets_dic[member]}
                for par in cfg['param_fixed']:
                    if not par in list(cluster_parsets_dic[member].keys()):
                        print("Setting fixed par ",par)
                        model_dics[f"{prefix}_ms{member}"][par] = cfg['param_fixed'][par]
                
                #OSHD specific fixed parameters
                model_dics[f"{prefix}_ms{member}"]['DD'] = 'static'
                model_dics[f"{prefix}_ms{member}"]['sfcf'] = 0
                model_dics[f"{prefix}_ms{member}"]['masswasting'] = False
                model_dics[f"{prefix}_ms{member}"]['petcf_seasonal'] = True
                model_dics[f"{prefix}_ms{member}"][cfg['OSHD']] = True
        
                print(model_dics[f"{prefix}_ms{member}"])
        
        elif calibration_purpose == 'Naive':
            prefix = f"{calibration_purpose}_{cfg['EXP_ID']}_{cfg['BASIN']}"
            for member in range(len(cluster_parsets_dic)):
                model_dics[f"{prefix}_ms{member}"] = {**cluster_parsets_dic[member]}
                for par in cfg['param_fixed']:
                    if not par in list(cluster_parsets_dic[member].keys()):
                        print("Setting fixed par ",par)
                        model_dics[f"{prefix}_ms{member}"][par] = cfg['param_fixed'][par]
                
                #OSHD specific fixed parameters
                model_dics[f"{prefix}_ms{member}"]['sfcf'] = 1.2
                model_dics[f"{prefix}_ms{member}"]['sfcf_scale'] = 1
                model_dics[f"{prefix}_ms{member}"]['DD_max'] = 4
                model_dics[f"{prefix}_ms{member}"]['TT'] = 0.0
                model_dics[f"{prefix}_ms{member}"]['tt_scale'] = 0 
                model_dics[f"{prefix}_ms{member}"]['rfcf'] = 1

                print(model_dics[f"{prefix}_ms{member}"])

   
      
    t0 = time.time()


    if cfg['SETTING'] == 'Synthetic':
        synobs = join(cfg['SYNDIR'],f"Q_{cfg['EXP_NAME']}_{cfg['OBS_ERROR']['scale']}.csv")
    
    else:
        synobs = None



    wflow = RunWflow_Julia(
        ROOTDIR = cfg['ROOTDIR'],
        PARSETDIR = cfg['EXPDIR'],
        BASIN = cfg['BASIN'],
        RUN_NAME = cfg['EXP_ID'],
        START_YEAR = start_year,
        END_YEAR = end_year,
        CONFIG_FILE = "sbm_config_CH_orig.toml",
        RESOLUTION = cfg['RESOLUTION'],
        YEARLY_PARAMS=False,
        SYNTHETIC_OBS=synobs,
        CONTAINER = cfg['CONTAINER'],
        NC_STATES=cfg['NC_STATES'])
    forcing_name = f"wflow_MeteoSwiss_{wflow.RESOLUTION}_{cfg['BASIN']}_{start_year-1}_{end_year}.nc"
    staticmaps_name = f"staticmaps_{wflow.RESOLUTION}_{cfg['BASIN']}_feb2024.nc"
    wflow.generate_MS_forcing_if_needed(forcing_name)
    wflow.check_staticmaps(staticmaps_name)
    # wflow.load_state(spinup_parsetname)
    model_dics_path = join(cfg['OUTDIR'],f"model_dics_{calibration_purpose}.json")
    with open(model_dics_path, "w") as f:
        json.dump(model_dics, f)
    wflow.load_model_dics(model_dics)
    wflow.adjust_config()
    # wflow.create_single_model('default_test9')
    wflow.create_models()
    initialize_time = time.time()-t0
    print(f"Initialization takes {initialize_time} seconds")

    # wflow.series_runs(wflow.degree_day_runs,  test = False )
    wflow.series_runs(wflow.standard_run, test = False)
    # wflow.parallel_runs(wflow.standard_run, test = False)
    wflow.finalize_runs()
    # wflow.save_state(list(model_dics.keys())[0],parsetname)
    wflow.load_stations([cfg['BASIN']])
    wflow.load_Q()
    wflow.stations_combine()
    # wflow.standard_plot(skip_first_year = True)
    wflow.station_OFs(skip_first_year=True)
    # print('combined mean',wflow.stations['Dischma'].combined.mean())
    # if calibration_purpose == 'Soilcalib':
    #     wflow.save_postrun(FOLDER,only_Q = True)
    # else:
    wflow.save_postrun(FOLDER,vars = cfg['NC_STATES'])
    # wflow.save_results(f"{RUN_NAME}_{cfg['BASIN']}")

    plt.show()
    time1 = time.time()- t0

    print(f"Years altogether takes {time1} seconds ({time1/60}) minutes")                  


        