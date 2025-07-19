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

os.chdir("/home/pwiersma/scratch/Scripts/Github/ewc")

from spotpy_calib import *
# from Yearlycalib import Yearlycalib
from spotpy_analysis import spotpy_analysis
# from Postruns import Postruns
# from Analysis import Analysis
from Synthetic_obs import Synthetic_obs
from Postruns import *
from Evaluation import *


#TODO Save config file to json in folder
#%% SETTINGS

# EXP_ID = 'Syn_209'
# EXP_ID = 'Syn_1000m_2014_2020_4000N_ROPE_noksatver'

# EXP_ID = 'Real_1000m_2014_2020_500N_DDS'
EXP_ID = 'Real_1000m_2014_2020_5000N_ROPE_288'
# EXP_ID = 'Syn_1000m_2014_2020_10N_MC'
BASIN = 'Dischma'
overwrite = False
postruns_only = False
TEST = True

# EXP_ID = 'Realtest'
json_file = join("/home/pwiersma/scratch/Data/ewatercycle/experiments/config_files",f"{EXP_ID}_config.json")
def check_for_files(dir,string):
    file_list = os.listdir(dir)
    for file in file_list:
        if string in file:
            return True
    return False

if os.path.isfile(json_file) and not overwrite:
    print("Config exists already")
    with open(json_file, 'r') as f:
        config = json.load(f)
        config['BASIN'] = BASIN	
else:
    config = dict(
        BASIN = BASIN,
        START_YEAR = 2014,
        END_YEAR = 2020,
        EXP_ID = EXP_ID,
        RESOLUTION = '1000m',
        OSHD = 'TI', # 'EB' or 'TI,
        TEST = TEST,

        SETTING = 'Synthetic', # 'Real' or 'Synthetic,
        OBS_ERROR = dict(scale = 0.1, 
                    kind = 'multiplicative', #additive or multiplicative
                    repeat = False),
        SYN_SNOWMODEL = 'OSHD', #or OSHD or wflow
        PERFECT_SOIL = True,

        #Calib settings
        ALGO = 'ROPE',
        OF = 'kge',
        OF_range = 0.02,
        N = 5000,
        PCT_FIRST_RUNS = 0.10, #actually a percentage
        # PCT = 0.1, 
        SUBSETS = 7,
        SOIL_K = 5,
        YEARLY_K = 5,
        DOUBLE_MONTHS = [4,5,6,7], #only valid for yearly stuff
        ROUTING_MARGIN = True,

        SNOWMONTHS = None,
        SNOWFREEMONTHS = None ,#[8,9,10,11,12,1,2,3],
        PEAK_FILTER = True,
        PBIAS_THRESHOLD = None,

        POSTERIOR_PCT = 0.5, #Percetage to keep for posterior "distribution"
    )
    if config['TEST'] == True:
        config['END_YEAR'] = config['START_YEAR']
        config['SOIL_K'] = 2
        config['YEARLY_K'] = 2
        config['N'] = 6
        config['ALGO'] = 'MC'
        config['OF_range'] = None
    if config['PERFECT_SOIL'] == True:
        config['SOIL_K'] = 1

    EXP_NAME = f"{config['EXP_ID']}_{config['BASIN']}_{config['SETTING']}"

    ROOTDIR = "/home/pwiersma/scratch/Data/ewatercycle/"
    EXPDIR = join(ROOTDIR, 'experiments')
    OUTDIR = join(ROOTDIR, 'outputs',f"{EXP_ID}")
    YEARLYDIR = join(OUTDIR, 'Yearlycalib')
    SOILDIR = join(OUTDIR, 'Soilcalib')
    CONTAINER = join(EXPDIR, 'containers',f"{EXP_ID}")

    Path(CONTAINER).mkdir(parents = True, exist_ok=True)             
    Path(OUTDIR).mkdir(parents = True, exist_ok=True)

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
    param_ranges = dict( KsatHorFrac = [1,500],#[180,220],#,#, #1,500
                        f = [0.1,10],#[3.5,4.5],#,#,  #0.1,10
                        # KsatVer = [0.1,5],
                        c= [1,20],#[1.5,2.5],#,#, #1,20
                        sfcf = [0.7,1.5],
                        sfcf_scale = [0.7,1.5],
                        rfcf = [0.7,1.5],
                        # DD_max = [3,7],
                        m_hock = [1,4],
                        r_hock = [0.005,0.04],
                        TT = [-1,1],
                        # tt_scale = [-0.5,0.5],
                        )
    param_kinds = dict( KsatHorFrac = 'soil',
                        f = 'soil',
                        # KsatVer = 'soil',
                        c= 'soil',
                        sfcf = 'meteo',
                        sfcf_scale = 'meteo',
                        rfcf = 'meteo',
                        # DD_max = 'snow',
                        m_hock = 'snow',
                        r_hock = 'snow',
                        TT = 'meteo',
                        tt_scale = 'meteo',
                        )
    param_soilcalib = dict( KsatHorFrac = True,
                            f = True,
                            # KsatVer = True,
                            c= True,
                            sfcf = True,
                            sfcf_scale = False,
                            rfcf = True,
                            # DD_max = True,
                            m_hock = True,
                            r_hock = False,
                            TT = False,
                            tt_scale = False,
                            )

    param_fixed = dict(CV = 0.3,
                        DD_min = 2,
                        m_hock = 2,
                        r_hock = 0.015,
                        WHC = 0.2,
                        mwf = 0.5,
                        vegcf = -0.5,
                        InfiltCapSoil = 1.0,
                        petcf_seasonal = True,
                        masswasting = True,
                        DD = 'Hock')

    config['param_ranges'] = param_ranges
    config['param_fixed'] = param_fixed
    config['param_kinds'] = param_kinds
    config['param_soilcalib'] = param_soilcalib
    # config['soil_extra_params'] = ['sfcf','rfcf','DD_max','m_hock']







# List all the files that are needed, check if they exist and copy them to temp folder

    # station_number = SwissStation(config['BASIN']).number

    # FILES_DICT = dict(
    # STATICMAPS_FILE = join(config['ROOTDIR'],"wflow_staticmaps",f"staticmaps_{config['RESOLUTION']}_{config['BASIN']}_feb2024.nc"),
    # FORCING_FILE = join(config['ROOTDIR'],"wflow_Julia_forcing",f"wflow_MeteoSwiss_{config['RESOLUTION']}_{config['BASIN']}_{config['START_YEAR']-1}_{config['END_YEAR']}.nc"),
    # DISCHARGE_FILE = glob.glob(join(config['ROOTDIR'],'Discharge_data','*'+str(station_number)+"*"))[0],
    # OSHD_file = join("/home/pwiersma/scratch/Data/SLF/OSHD",f"OSHD_{config['RESOLUTION']}_latlon_{config['BASIN']}.nc"),
    # IPOT_file = join(config['ROOTDIR'],"aux_data","Ipot",f"Ipot_DOY_{config['RESOLUTION']}_{config['BASIN']}.nc"),
    # CONFIG_FILE = json_file,
    # PET_COMPARISON_FILE = join(config['ROOTDIR'],"pet_comparison_monthly.csv"))

    #TODO change OSHD dir inRunfwlow_Julia
    #ewatercycle config file in Runwlflow_Julia


    # DISCHARGE_FILE 
    # OSHD_FILE
    # IPOT_FILE
    # CONFIG_FILE
    # PET_COMPARISON_FILE
    #DHM_25_FILE
    #RIVERS_FILE






    #%% Make synthetic obs if necessary
    if config['SETTING'] == 'Synthetic':
        SYNDIR = join(config['OUTDIR'], 'Synthetic_obs')

        trueparams =  dict( KsatHorFrac = 200,
                                f = 4.0,
                                thetaR = 1.0,
                                thetaS = 1.0,
                                    # KsatVer = 1.0,
                                    # SoilThickness = 1.0,
                                    InfiltCapSoil = 1.0,
                                    c = 2,
                                    sfcf_start = 0.9,
                                    sfcf_change = 0.05,
                                    # sfcf_scale = 1.2,
                                    rfcf = 1.2,
                                    # DD_min = 2.0,
                                    # DD_max = 6.0,
                                    m_hock = 2,
                                    r_hock = 0.015,
                                    mwf = 0.5,
                                    WHC = 0.2,
                                    TT = 0.0,
                                    # tt_scale = 0.5,
                                    vegcf = -0.5,
                                    CV = 0.3 
                                    )
        if config['SYN_SNOWMODEL'] == 'wflow':
            print('wflow snow model with yearly sfcf')
            sfcf_start = trueparams['sfcf_start']
            direction = 1
            for year in range(config['START_YEAR'],config['END_YEAR']+1):
                trueparams[f"sfcf_{year}"] = np.round(sfcf_start,4)
                if trueparams[f"sfcf_{year}"] in  config['param_ranges']['sfcf']:
                    direction *= -1
                sfcf_start += trueparams['sfcf_change']*direction
                print(sfcf_start)
            #remove sfcf_start and sfcf_change from trueparams
            trueparams.pop('sfcf_start')
            trueparams.pop('sfcf_change')

            print("True params:",trueparams)

        config['SYNDIR'] = SYNDIR
        config['trueparams'] = trueparams

        if os.path.exists(SYNDIR) and check_for_files(SYNDIR,'Q_') and check_for_files(SYNDIR,'SWE_'):
                print("Synthetic done")
        else:
            # RUN_NAME = f"{config['EXP_ID']}_{config['BASIN']}_Synthetic_obs"
            RUN_NAME = f"Synthetic_obs"

            Synthetic_obs(config, RUN_NAME)
            #TODO Make synthetic Q with OSHD input 


        #Write config file to json
    with open(json_file, 'w') as f:
        json.dump(config, f)
    #%% Soil Calib
    # SOILRUN_NAME = f"{config['EXP_ID']}_{config['BASIN']}_Soilcalib"

    print("Checking files")
    
    CONTAINER = join(EXPDIR, 'containers',f"{EXP_ID}")

    Path(CONTAINER).mkdir(parents = True, exist_ok=True) 
    config['CONTAINER'] = CONTAINER

    station_number = SwissStation(config['BASIN']).number

    orig_files_dict = dict(
    STATICMAPS_FILE = join(config['ROOTDIR'],"experiments/Data/input",f"staticmaps_{config['RESOLUTION']}_{config['BASIN']}_feb2024.nc"),
    FORCING_FILE = join(config['ROOTDIR'],"experiments/Data/input",
        f"wflow_MeteoSwiss_{config['RESOLUTION']}_{config['BASIN']}_{config['START_YEAR']-2}_{config['END_YEAR']}.nc"),
    # STATICMAPS_FILE = f"staticmaps_{config['RESOLUTION']}_{config['BASIN']}_feb2024.nc"),
    # FORCING_FILE = f"wflow_MeteoSwiss_{config['RESOLUTION']}_{config['BASIN']}_{config['START_YEAR']-1}_{config['END_YEAR']}.nc"),
    DISCHARGE_FILE = glob.glob(join(config['ROOTDIR'],'Discharge_data','*'+str(station_number)+"*"))[0],
    OSHD_file = join("/home/pwiersma/scratch/Data/SLF/OSHD",f"OSHD_{config['RESOLUTION']}_latlon_{config['BASIN']}.nc"),
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



if config['PERFECT_SOIL'] == False:
        
    SOILRUN_NAME = f"{config['BASIN']}_Soilcalib_{config['EXP_ID']}"

    if os.path.exists(config['SOILDIR']) and (os.path.isfile(join(config['SOILDIR'],f"{SOILRUN_NAME}.csv"))):
    # if os.path.isfile(SOILFILE):
        print("Soil calib done")
    else:
        Path(config['SOILDIR']).mkdir(parents = True, exist_ok=True)
        SoilCalib(config,SOILRUN_NAME)

    clusterfile = join(config['SOILDIR'],f"{config['BASIN']}_ksoil{config['SOIL_K']}_{'posterior'}.csv")
    if os.path.isfile(clusterfile):
        print("Spotpy soil analysis done")
    else:
        SA  = spotpy_analysis(config, SOILRUN_NAME, 'Soilcalib')



elif config['PERFECT_SOIL'] and config['SETTING'] == 'Synthetic':
    Path(config['SOILDIR']).mkdir(parents = True, exist_ok=True)
    SOILRUN_NAME = f"{config['BASIN']}_Soilcalib_{config['EXP_ID']}"
    for p in ['prior','posterior']:
        file = f"{config['BASIN']}_ksoil{config['SOIL_K']}_{p}.csv"
        #select all parameters for which params_soilcalib is true
        perfect_soilpars = pd.DataFrame()
        for param in config['param_soilcalib']:
            if config['param_soilcalib'][param]:
                if param =='sfcf':
                    perfect_soilpars[param] = 1
                else:
                    perfect_soilpars[param] = [config['trueparams'][param]]
        perfect_soilpars.to_csv(join(config['SOILDIR'],file))
    



#%% Spotpy soil calib


#%% Postruns
if (os.path.isfile(join(config['SOILDIR'],f"{config['BASIN']}_Q.csv"))) and (postruns_only == False):
    print("Postruns done")
else:   
    Postruns(config,'Soilcalib')    

# OSHD and naive postruns 
if config['SETTING']=='Real':
    #OSHD
    # if (os.path.isfile(join(config['OUTDIR'], 'OSHD',f"{config['BASIN']}_snow.nc")))and (postruns_only == False):
    #     print("OSHD done")
    # else:
    #     Path(join(config['OUTDIR'], 'OSHD')).mkdir(parents = True, exist_ok=True)
    #     Postruns(config, 'OSHD')

    #OSHD with yearly rfcf 
    #we just take one, no ensemble needed? 
    if os.path.isfile(join(config['OUTDIR'], 'OSHD',str(config['START_YEAR']),"0",f"{config['BASIN']}_{config['START_YEAR']}_ms0_OB_calib.csv")):
        print("OSHD yearly done")
    else:
        for year in range(config['START_YEAR'], config['END_YEAR'] + 1):
            for ksoil in range(config['SOIL_K']):
                OB_calib(config,year,ksoil)

    #collect rfcfs and do some postruns 
    #the difference with yearlycalib is that we have 5 param sets per year, instead of 25
    #there is no kyearly, so we can do one postrun per year 
    if (os.path.isfile(join(config['OUTDIR'], 'OSHD',f"{config['START_YEAR']}",f"{config['BASIN']}_Q.csv")))and (postruns_only == False):
        print("OSHD postruns done")
    else:
        for year in range(config['START_YEAR'], config['END_YEAR'] + 1):
            Postruns(config, 'OB_calib', year)
    # sys.exit()


    
    #Naive
    if (os.path.isfile(join(config['OUTDIR'], 'Naive',f"{config['BASIN']}_snow.nc")))and (postruns_only == False):
        print("Naive done")
    else:
        Path(join(config['OUTDIR'], 'Naive')).mkdir(parents = True, exist_ok=True)
        Postruns(config, 'Naive')

#%% Yearly calib 
#check if there are soil_k spotpy files in each yearly folder 
if config['TEST'] == True:
    file_list = [join(config['YEARLYDIR'], 
                    f"{config['START_YEAR']}",
                            "0",
                    f"{config['BASIN']}_{config['START_YEAR']}_ms0.csv")]
else:
    file_list = [
        join(join(config['YEARLYDIR'], f"{year}"),f"{k}", f"{config['BASIN']}_{year}_ms{k}_{config['EXP_ID']}.csv")
        for year in range(config['START_YEAR'], config['END_YEAR'] + 1)
        for k in range(config['SOIL_K'])
    ]

if np.all([os.path.isfile(file) for file in file_list]):
    print("Yearly calib done")
else:
    nofiles = [file for file in file_list if not os.path.isfile(file)]
    years = []
    for f in nofiles:
        # Split the path and extract the year
        parts = f.split('/')
        year = parts[-3]
        if not year in years:
            years.append(year)
        print(f"Year in file is {year}")

    #Call YearlyCalib_joint, which sends one job per year to the terminal
    #Each job launches one call of Yearlycalib per soil cluster
    Path(config['YEARLYDIR']).mkdir(parents = True, exist_ok=True)
    YearlyCalib_joint(config,years)

#%% Spotpy plots and clusters 
#Create figures for each year and for each soil parameter set
#Maybe make joint figures 
if config['TEST'] == True:
    yearly_spotpy_dirs = [join(config['YEARLYDIR'], f"{config['START_YEAR']}"),"0",f"{config['BASIN']}_Plots"]
else:
    yearly_spotpy_dirs = [join(config['YEARLYDIR'],
                                f"{year}",
                                f"{member}",
                                f"{config['BASIN']}_Plots") 
                                for year in range(config['START_YEAR'], config['END_YEAR'] + 1)
                                for member in range(config['SOIL_K'])]

if np.all([os.path.exists(dir) for dir in yearly_spotpy_dirs]):
    print("Yearly spotpy plots done")
else:
    for year in range(config['START_YEAR'], config['END_YEAR'] + 1):
        for k in range(config['SOIL_K']):
            YRUN_NAME = f"{config['BASIN']}_{year}_ms{k}_{config['EXP_ID']}"
            spotpy_analysis(config,YRUN_NAME,
                                'Yearlycalib',
                            single_year = year,
                            member = k)
        # if config['TEST']:
        #     break


#%% Postruns
#Gather the k_soil*k_yearly best parameter sets and run for each year
#Merge all products into a multiyear product
if config['TEST']==True: 
    post_run_files = [join(config['YEARLYDIR'],
                            f"{config['START_YEAR']}", 
                            "0",
                            f"{config['BASIN']}_snow.nc")]
else:
    post_run_files = [join(config['YEARLYDIR'], 
                        str(year), 
                            str(ksoil),
                            f"{config['BASIN']}_snow.nc")
                    for year in range(config['START_YEAR'], config['END_YEAR'] + 1)
                    for ksoil in range(config['SOIL_K'])]

if (np.all([os.path.isfile(file) for file in post_run_files])) and (postruns_only == False):
    print("Postruns done")
else:
    for year in range(config['START_YEAR'], config['END_YEAR'] + 1):
        for ksoil in range(config['SOIL_K']):
            Postruns(config, 'Yearlycalib', year, ksoil)


                    

#%%
E = Evaluation(config)
E.load_Q()
E.load_OFs()

E.load_SWE()
E.load_SWEbands()
E.load_pars()
E.plot_pars()
E.load_scalaroutput()
E.compute_yearly_scalars()
E.compute_daily_scalars()
# E.plot_OF()
E.hydrographs_soilcalib()
E.hydrographs_yearlycalib()
# E.plot_swebands() 
# E.plot_waterbalance()
E.composite_plot()
if config['SETTING'] == 'Real':
    # E.plot_each_station()
    E.stations_OF()
    E.load_SC()
    E.plot_OA()




# #%% Analysis
#Analyze everything 


# ANALYSISDIR = join(OUTDIR, 'Analysis')
# if os.path.exists(ANALYSISDIR):
#     print("Analysis done")
# else:
#     Analysis(config)


# %%





