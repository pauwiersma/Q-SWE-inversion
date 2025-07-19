#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:00:00 2021

@author: Pau Wiersma


"""
#%%


import os
from os.path import join
from pathlib import Path
import json
import shutil
import sys

from spotpy_calib import *
from spotpy_analysis import *
from Synthetic_obs import *
from Postruns import *
from Evaluation import *
from generate_forcing import generate_MS_forcing
import argparse

ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')
os.chdir(ROOTDIR)

#TODO Save config file to json in folder
#%% SETTINGS
# EXP_ID = 'Syn_test1'
# BASIN = 'Dischma'
# TEST = False
# postruns_only = False

def file_exists_in_directory(dir, string):
# def check_for_files(dir,string):
    file_list = os.listdir(dir)
    for file in file_list:
        if string in file:
            return True
    return False
def preproc(BASIN,EXP_ID,  TEST = False, postruns_only = False):     

    # EXP_ID = 'Syn_1000m_2014_2020_4000N_ROPE_noksatver'

    # EXP_ID = 'Real_1000m_2014_2020_500N_DDS'
    # EXP_ID = 'Real_1000m_2014_2020_5000N_ROPE_288'
    # EXP_ID = 'Syn_1000m_2014_2020_10N_MC'
    overwrite = True

    # EXP_ID = 'Realtest'
    # json_file = join("/home/pwiersma/scratch/Data/ewatercycle/experiments/config_files",f"{EXP_ID}_config.json")
    
    ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')
    config_dir = join(ROOTDIR, 'experiments', 'config_files')
    # config_dir = os.getenv('CONFIG_DIR', default = '/home/pwiersma/scratch/Data/ewatercycle/experiments/config_files')
    json_file = join(config_dir, f"{EXP_ID}_config.json")
    if os.path.isfile(json_file) and not overwrite:
        print("Config exists already")
        with open(json_file, 'r') as f:
            config = json.load(f)
            config['BASIN'] = BASIN	
    else:
        config = dict(
            BASIN = BASIN,
            START_YEAR = 2001,
            END_YEAR = 2022,
            EXP_ID = EXP_ID,
            RESOLUTION = '1000m',
            OSHD = 'TI', # 'EB' or 'TI,

            SETTING = 'Synthetic', # 'Real' or 'Synthetic,
            OBS_ERROR = dict(scale =0.0,# 0.1, 
                        kind = 'multiplicative', #additive or multiplicative
                        repeat = False),
            SYN_SNOWMODEL = 'Hock', #or Hock or seasonal or OSHD
            PERFECT_SOIL = True, #if True, only one soil parameter set is used and step 1 is skipped 
            SWE_CALIB = False, #calibrate on SWE instead of discharge
            YEARLY_SOIL = False, #adds soil parameters to yearly calibration
            LOWER_BENCHMARK = False, #if true, fix forcing parameters to default value
        #     NC_STATES =['snow','snowwater','actevap','leakage','rfmelt','canopystorage','interception'
        # ,'satstore','unsatstore','canopy_evap_fraction','throughfall','runoff',
        # 'Q_river','Qin_river','inwater_river','Q_land','Qin_land','inwater_land',
        # 'to_river_land','ssf','ssfin','to_river_ssf',
        #                 'transfer','vwc','zi','unsatstore_layer','act_thickl'], 
            NC_STATES =['snow','snowwater','runoff'],#,'actevap','actleakage','rainfallplusmelt','canopystorage','interception'
        # ,'satwaterdepth','ustoredepth','e_r','throughfall','runoff',
        # 'q_river','qin_river','inwater_river','q_land','qin_land','inwater_land',
        # 'to_river_land','ssf','ssfin','to_river_ssf',
        #                 'transfer','vwc','zi','ustorelayerdepth','act_thickl'],
            SAVE_NC_STATES_SPOTPY = True, 
            #Calib settings
            # ALGO = 'ROPE',
            ALGO = 'LHS',#'LHS',
            SAVE_SPOTPY_SWE = True, 
            PARALLEL = True,  #seq or mpi
            OF = 'kge',
            OF_range = 0.02,
            N = 10000,
            PCT_FIRST_RUNS = 0.10, #actually a percentage
            # PCT = 0.1, 
            SUBSETS = 7,
            SOIL_K = 5,
            YEARLY_K = 5,
            DOUBLE_MONTHS = [4,5,6,7], #only valid for yearly stuff
            ROUTING_MARGIN = True,

            SNOWMONTHS = None,
            SNOWFREEMONTHS = None ,#[8,9,10,11,12,1,2,3],
            PEAK_FILTER = False,
            PBIAS_THRESHOLD = None,

            POSTERIOR_PCT = 0.5, #Percetage to keep for posterior "distribution"
        )
        if TEST:
            print("Running in test mode")
            config['TEST'] = True
            config['END_YEAR'] = config['START_YEAR'] #+ 1
            config['SOIL_K'] = 2
            config['YEARLY_K'] = 2
            config['N'] = 6
            config['ALGO'] = 'MC'
            config['OF_range'] = None
            # config['PARALLEL'] = False
        if config['PERFECT_SOIL'] == True:
            config['SOIL_K'] = 1
        if config['LOWER_BENCHMARK']:
            config['N'] /= 2
        EXP_NAME = f"{config['EXP_ID']}_{config['BASIN']}_{config['SETTING']}"

        EXPDIR = join(ROOTDIR, 'experiments')
        OUTDIR = join(ROOTDIR, 'outputs',f"{EXP_ID}")
        YEARLYDIR = join(OUTDIR, 'Yearlycalib')
        SOILDIR = join(OUTDIR, 'Soilcalib')
        CONTAINER = join(EXPDIR, 'containers',f"{EXP_ID}")
                    
        Path(OUTDIR).mkdir(parents = True, exist_ok=True)
        Path(CONTAINER).mkdir(parents = True, exist_ok=True)

        config['EXP_NAME'] = EXP_NAME
        config['ROOTDIR'] = ROOTDIR
        config['EXPDIR'] = EXPDIR
        config['OUTDIR'] = OUTDIR
        config['SOILDIR'] = SOILDIR
        config['YEARLYDIR'] = YEARLYDIR
        config['CONTAINER'] = CONTAINER

        # param_ranges = dict(soil = dict( KsatHorFrac = [1,1000],
        #                                 f = [0.1,10],
        #                                 KsatVer = [0.1,10],
        #                                 c= [1,20]),
        #                     snow = dict( DD_max = [3,7]),
        #                     meteo = dict( sfcf = [0.7,1.5],
        #                                  sfcf_scale = [0.7,1.5],
        #                                 rfcf = [0.7,1.5],
        #                                 TT = [-1,1],
        #                                 tt_scale = [-0.5,0.5],
        #                                 )
        #                     )
        param_ranges = dict( KsatHorFrac = [75,125],#[1,50],#[1,500],
                            f = [0.8,1.2],#[0.1,10],
                            # KsatVer = [0.1,5],
                            # c= [0.5,10],
                            sfcf = [0.9,1.5],
                            sfcf_scale = [0.7,1.3],
                            rfcf = [0.7,1.3],
                            # DD_min = [1,4],
                            # DD_max = [3,7],
                            m_hock = [1,4],
                            r_hock = [0.01,0.04],
                            TT = [-1,1],
                            tt_scale = [-0.5,0.5],
                            mwf = [0.1,0.9],
                            WHC = [0.1,0.4],
                            # vegcf = [-1,1],
                            CV = [0.1,0.5]
                            )
        

        param_kinds = dict( KsatHorFrac = 'soil',
                            f = 'soil',
                            # KsatVer = 'soil',
                            c= 'soil',
                            sfcf = 'meteo',
                            sfcf_scale = 'meteo',
                            rfcf = 'meteo',
                            # DD_min = 'snow',
                            # DD_max = 'snow',
                            m_hock = 'snow',
                            r_hock = 'snow',
                            TT = 'meteo',
                            tt_scale = 'meteo',
                            mwf = 'snow',
                            WHC = 'snow',
                            vegcf = 'snow',
                            CV = 'snow'
                            )
        if config['YEARLY_SOIL']:
            param_kinds = {key: 'snow' if value == 'soil' else value for key, value in param_kinds.items()}
        param_soilcalib = dict( KsatHorFrac = True,
                                f = True,
                                # KsatVer = True,
                                c= True,
                                sfcf = True,
                                sfcf_scale = False,
                                rfcf = True,
                                # DD_min = False,
                                # DD_max = False,
                                m_hock = False,
                                r_hock = False,
                                TT = False,
                                tt_scale = False,
                                mwf = False,
                                WHC = False,
                                vegcf = False,
                                CV = False,
                                )

        param_fixed = dict(CV = 0.3,
                            DD_min = 2,
                            DD_max = 4,
                            m_hock = 2,
                            r_hock = 0.015,
                            WHC = 0.2,
                            mwf = 0.5,
                            # vegcf = -0.5,
                            InfiltCapSoil = 1.0,
                            rfcf = 1.0,
                            sfcf = 1.2, 
                            sfcf_scale = 1.0,
                            tt_scale = 0.0,
                            petcf_seasonal = True,
                            masswasting = True,
                            DD = 'Hock',
                            T_offset = 0)

        config['param_ranges'] = param_ranges
        config['param_fixed'] = param_fixed
        config['param_kinds'] = param_kinds
        config['param_soilcalib'] = param_soilcalib
        # config['soil_extra_params'] = ['sfcf','rfcf','DD_max','m_hock']


#%% Synthetic settings
        if config['SETTING'] == 'Synthetic':
            SYNDIR = join(config['OUTDIR'], 'Synthetic_obs')
            trueparams =  dict( KsatHorFrac = 100,
                                    f = 1.0,
                                    thetaR = 1.0,
                                    thetaS = 1.0,
                                        KsatVer = 1,
                                        # SoilThickness = 1.0,
                                        InfiltCapSoil = 1.0,
                                        c = 1,
                                        sfcf_start = 1.0,
                                        sfcf_change = 0.1,
                                        sfcf_scale = 1.0,
                                        rfcf = 1,
                                        DD_min = 2.0,
                                        DD_max = 6.0,
                                        m_hock = 2.5,
                                        r_hock = 0.025,
                                        mwf = 0.5,
                                        WHC = 0.25,
                                        TT = 0.0,
                                        tt_scale = 0.0,
                                        # vegcf = -0.5,
                                        CV = 0.3 
                                        )
            
            sfcf_start = trueparams['sfcf_start']
            direction = 1
            for year in range(config['START_YEAR'],config['END_YEAR']+1):
                trueparams[f"sfcf_{year}"] = np.round(sfcf_start,4)
                sfcf_start += trueparams['sfcf_change'] * direction
                bounds = config['param_ranges']['sfcf']
                sfcf_min = bounds[0]
                sfcf_max = bounds[1]
                # Reverse direction if next step would exceed bounds
                if sfcf_start + (trueparams['sfcf_change'] * direction) >= sfcf_max or sfcf_start + (trueparams['sfcf_change'] * direction) <= sfcf_min:
                    direction *= -1

                # if trueparams[f"sfcf_{year}"] in  config['param_ranges']['sfcf']:
                #     if year != config['START_YEAR']:
                #         direction *= -1
                # sfcf_start += trueparams['sfcf_change']*direction
                print(sfcf_start)
            #remove sfcf_start and sfcf_change from trueparams
            trueparams.pop('sfcf_start')
            trueparams.pop('sfcf_change')

            print("True params:",trueparams)

            # trueparams =  dict( KsatHorFrac = 100,
            #                         f = 1.0,
            #                         thetaR = 1.0,
            #                         thetaS = 1.0,
            #                             KsatVer = 1,
            #                             # SoilThickness = 1.0,
            #                             InfiltCapSoil = 1.0,
            #                             c = 1,
            #                             sfcf_start = 0.9,
            #                             sfcf_change = 0.1,
            #                             sfcf_scale = 1.0,
            #                             rfcf = 1,
            #                             DD_min = 2.0,
            #                             DD_max = 6.0,
            #                             m_hock = 2.5,
            #                             r_hock = 0.025,
            #                             mwf = 0.5,
            #                             WHC = 0.25,
            #                             TT = 0.0,
            #                             tt_scale = 0.0,
            #                             # vegcf = -0.5,
            #                             CV = 0.3 
            #                             )
            
            # sfcf_start = trueparams['sfcf_start']
            # direction = 1
            # for year in range(config['START_YEAR'],config['END_YEAR']+1):
            #     trueparams[f"sfcf_{year}"] = np.round(sfcf_start,4)
            #     if trueparams[f"sfcf_{year}"] in  config['param_ranges']['sfcf']:
            #         if year != config['START_YEAR']:
            #             direction *= -1
            #     sfcf_start += trueparams['sfcf_change']*direction
            #     print(sfcf_start)
            # #remove sfcf_start and sfcf_change from trueparams
            # trueparams.pop('sfcf_start')
            # trueparams.pop('sfcf_change')

            # print("True params:",trueparams)

            config['SYNDIR'] = SYNDIR
            config['trueparams'] = trueparams
        if config['LOWER_BENCHMARK']:
                #pop sfcf, rfcf, sfcf_scale and tt_Scale from param_ranges
                param_ranges.pop('sfcf', None)
                param_ranges.pop('rfcf', None)
                param_ranges.pop('sfcf_scale', None)
                param_ranges.pop('TT', None)
                param_ranges.pop('tt_scale', None)
    
        #Write config file to json
        with open(json_file, 'w') as f:
            json.dump(config, f)

    #%% List all the files that are needed, check if they exist and copy them to temp folder
    print("Checking files")
    station_number = SwissStation(config['BASIN']).number

    orig_files_dict = dict(
    STATICMAPS_FILE = join(config['ROOTDIR'],"experiments/data/input",f"staticmaps_{config['RESOLUTION']}_{config['BASIN']}_feb2024.nc"),
    FORCING_FILE = join(config['ROOTDIR'],"experiments/data/input",
        f"wflow_MeteoSwiss_{config['RESOLUTION']}_{config['BASIN']}_{config['START_YEAR']-2}_{config['END_YEAR']}.nc"),
    # STATICMAPS_FILE = f"staticmaps_{config['RESOLUTION']}_{config['BASIN']}_feb2024.nc"),
    # FORCING_FILE = f"wflow_MeteoSwiss_{config['RESOLUTION']}_{config['BASIN']}_{config['START_YEAR']-1}_{config['END_YEAR']}.nc"),
    DISCHARGE_FILE = glob.glob(join(config['ROOTDIR'],'Discharge_data','*'+str(station_number)+"*"))[0],
    # OSHD_file = join("/home/pwiersma/scratch/Data/SLF/OSHD",f"OSHD_{config['RESOLUTION']}_latlon_{config['BASIN']}.nc"),
    OSHD_file = join(config['ROOTDIR'],"OSHD",f"OSHD_{config['RESOLUTION']}_latlon_{config['BASIN']}.nc"),
    IPOT_file = join(config['ROOTDIR'],"aux_data","Ipot",f"Ipot_DOY_{config['RESOLUTION']}_{config['BASIN']}.nc"),
    # CONFIG_FILE = json_file,
    PET_COMPARISON_FILE = join(config['ROOTDIR'],"aux_data","pet_comparison_monthly.csv"))

    IS_SM = os.path.isfile(orig_files_dict['STATICMAPS_FILE'])
    IS_FORCING = os.path.isfile(orig_files_dict['FORCING_FILE'])
    if not IS_SM or not IS_FORCING:
        wflow_object = RunWflow_Julia(
                    ROOTDIR = config['ROOTDIR'],
                    PARSETDIR = config['EXPDIR'],
                    BASIN = config['BASIN'],
                    RUN_NAME = config['EXP_ID'],
                    START_YEAR = config['START_YEAR']-1,
                    END_YEAR = config['END_YEAR'],
                    CONFIG_FILE = "sbm_config_CH_orig.toml",
                    RESOLUTION = config['RESOLUTION'],
                    YEARLY_PARAMS=False,
                    SYNTHETIC_OBS=config['OBS_ERROR']['scale'],
                    CONTAINER= config['CONTAINER'])
        wflow_object.check_staticmaps(os.path.basename(orig_files_dict['STATICMAPS_FILE']))
        wflow_object.generate_MS_forcing_if_needed(os.path.basename(orig_files_dict['FORCING_FILE']))

    new_files_dict = dict()
    for key in orig_files_dict:
        new_files_dict[key] = join(config['CONTAINER'],os.path.basename(orig_files_dict[key]))
        
        if not os.path.isfile(new_files_dict[key]):
            if os.path.isfile(orig_files_dict[key]):
                shutil.copy(orig_files_dict[key],new_files_dict[key])
            else:
                print(f"File {orig_files_dict[key]} does not exist")
                sys.exit()
                

    #ewatercycle config file in Runwlflow_Julia


    if config['SETTING'] == 'Synthetic':
        # if os.path.exists(SYNDIR) and file_exists_in_directory(SYNDIR,'Q_') and file_exists_in_directory(SYNDIR,'SWE_'):
        if os.path.exists(SYNDIR) and file_exists_in_directory(SYNDIR,'Q_') and file_exists_in_directory(SYNDIR,'SWE_'):
            print("Synthetic done")
        else:
            # RUN_NAME = f"{config['EXP_ID']}_{config['BASIN']}_Synthetic_obs"
            RUN_NAME = f"Synthetic_obs_{config['BASIN']}_{config['EXP_ID']}"

            Synthetic_obs(config, RUN_NAME).generate_synthetic_obs()
            #TODO Make synthetic Q with OSHD input 

            # S= Synthetic_obs(config, RUN_NAME)




    #%% Soil Calib
    # SOILRUN_NAME = f"{config['EXP_ID']}_{config['BASIN']}_Soilcalib"
    # if config['PERFECT_SOIL'] == False:

    #     SOILRUN_NAME = f"{config['BASIN']}_Soilcalib_{config['EXP_ID']}"

    #     if os.path.exists(config['SOILDIR']) and (os.path.isfile(join(config['SOILDIR'],f"{SOILRUN_NAME}.csv"))):
    #     # if os.path.isfile(SOILFILE):
    #         print("Soil calib done")
    #     else:
    #         Path(config['SOILDIR']).mkdir(parents = True, exist_ok=True)
    #         SoilCalib(config,SOILRUN_NAME)


    # #%% Spotpy soil calib
    #     clusterfile = join(config['SOILDIR'],f"{config['BASIN']}_ksoil{config['SOIL_K']}_{'posterior'}.csv")
    #     if os.path.isfile(clusterfile):
    #         print("Spotpy soil analysis done")
    #     else:
    #         SA  = spotpy_analysis(config, SOILRUN_NAME, 'Soilcalib')
    
    
    # elif config['PERFECT_SOIL'] and config['SETTING'] == 'Synthetic':
    #     print("Skipping soilcalib")
    #     Path(config['SOILDIR']).mkdir(parents = True, exist_ok=True)
    #     SOILRUN_NAME = f"{config['BASIN']}_Soilcalib_{config['EXP_ID']}"
    #     for p in ['prior','posterior']:
    #         file = f"{config['BASIN']}_ksoil{config['SOIL_K']}_{p}.csv"
    #         #select all parameters for which params_soilcalib is true
    #         perfect_soilpars = pd.DataFrame()
    #         for param in config['param_soilcalib']:
    #             if config['param_soilcalib'][param]:
    #                 if param =='sfcf':
    #                     perfect_soilpars[param] = 1
    #                 else:
    #                     perfect_soilpars[param] = [config['trueparams'][param]]
    #         perfect_soilpars.to_csv(join(config['SOILDIR'],file))
    


    # #%% Postruns
    # if (os.path.isfile(join(config['SOILDIR'],f"{config['BASIN']}_Q.csv"))) and (postruns_only == False):
    #     print("Postruns done")
    # else:   
    #     Postruns(config,'Soilcalib')    

    # # OSHD and naive postruns 
    # if config['SETTING']=='Real':
    #     #OSHD
    #     # if (os.path.isfile(join(config['OUTDIR'], 'OSHD',f"{config['BASIN']}_snow.nc")))and (postruns_only == False):
    #     #     print("OSHD done")
    #     # else:
    #     #     Path(join(config['OUTDIR'], 'OSHD')).mkdir(parents = True, exist_ok=True)
    #     #     Postruns(config, 'OSHD')

    #     #OSHD with yearly rfcf 
    #     #we just take one, no ensemble needed? 
    #     if os.path.isfile(join(config['OUTDIR'], 'OSHD',str(config['START_YEAR']),"0",f"{config['BASIN']}_{config['START_YEAR']}_ms0_OB_calib.csv")):
    #         print("OSHD yearly done")
    #     else:
    #         for year in range(config['START_YEAR'], config['END_YEAR'] + 1):
    #             for ksoil in range(config['SOIL_K']):
    #                 OB_calib(config,year,ksoil)

    #     #collect rfcfs and do some postruns 
    #     #the difference with yearlycalib is that we have 5 param sets per year, instead of 25
    #     #there is no kyearly, so we can do one postrun per year 
    #     if (os.path.isfile(join(config['OUTDIR'], 'OSHD',f"{config['START_YEAR']}",f"{config['BASIN']}_Q.csv")))and (postruns_only == False):
    #         print("OSHD postruns done")
    #     else:
    #         for year in range(config['START_YEAR'], config['END_YEAR'] + 1):
    #             Postruns(config, 'OB_calib', year)
    #     # sys.exit()


        
    #     #Naive
    #     if (os.path.isfile(join(config['OUTDIR'], 'Naive',f"{config['BASIN']}_snow.nc")))and (postruns_only == False):
    #         print("Naive done")
    #     else:
    #         Path(join(config['OUTDIR'], 'Naive')).mkdir(parents = True, exist_ok=True)
    #         Postruns(config, 'Naive')
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('BASIN', type=str, help='The basin identifier')
    parser.add_argument('EXP_ID', type=str, help='The experiment ID')
    # parser.add_argument('--test', dest='TEST', action='store_true', help='Run in test mode')
    parser.add_argument('--test', dest = 'TEST', type=str, choices=['true', 'false', '1', '0'], default='false', help='Run in test mode')

    args = parser.parse_args()

    BASIN = args.BASIN
    EXP_ID = args.EXP_ID
    if args.TEST in ['true', '1']:
        TEST = True
    else:
        TEST = False
    # EXP_ID = sys.argv[2]
    # BASIN = sys.argv[1]
    preproc(BASIN,EXP_ID, TEST=TEST)