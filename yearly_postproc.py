import os
from os.path import join
from pathlib import Path


RUNDIR = os.getenv('EWC_RUNDIR', default = '/home/pwiersma/scratch/Scripts/Github/ewc/')
os.chdir(RUNDIR)

from spotpy_calib import SoilCalib
from spotpy_analysis import spotpy_analysis
from Synthetic_obs import Synthetic_obs
from Postruns import *
from Evaluation import *
import argparse


#%% Yearly calib 
#check if there are soil_k spotpy files in each yearly folder 
EXP_ID = "Syn_1811"
BASIN = "Dischma"
TEST = False
postruns_only = False


def yearly_postproc(BASIN,EXP_ID,postruns_only = False,TEST = False):
    EWC_ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')
    CONFIG_DIR = join(join(EWC_ROOTDIR,"experiments",
        "config_files",f"{EXP_ID}_config.json"))
    config = json.load(open(CONFIG_DIR))
    config['BASIN'] = BASIN
    config['TEST'] = TEST


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
                                member = k).run()
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
                        # if config['TEST'] == True:
                        #         break

    #%%
    # config['CONTAINER'] = None
    # config['SWE_CALIB'] = None
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
    if not config['SWE_CALIB']:
        E.hydrographs_soilcalib()
        E.hydrographs_yearlycalib()

    ilats = [9,3,3,3]
    ilons = [2,2,6,10]
    # # ilat = 10
    # # ilon = 5
    timerange = slice(f'{config["START_YEAR"]-1}-10-01', f'{config["START_YEAR"]}-09-30')
    for ilat,ilon in zip(ilats,ilons):
        E.plot_pixel_scalars(ilat = ilat,ilon = ilon,timerange =timerange)
    # E.plot_swebands() 
    # E.plot_waterbalance()
    E.composite_plot()
    if config['SETTING'] == 'Real':
        # E.plot_each_station()
        E.stations_OF()
        E.load_SC()
        E.plot_OA()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('BASIN', type=str, help='The basin identifier')
    parser.add_argument('EXP_ID', type=str, help='The experiment ID')
    parser.add_argument('--test', dest = 'TEST', type=str, choices=['true', 'false', '1', '0'], default='false', help='Run in test mode')

    args = parser.parse_args()

    BASIN = args.BASIN
    EXP_ID = args.EXP_ID
    if args.TEST in ['true', '1']:
        TEST = True
    else:
        TEST = False
    yearly_postproc(BASIN,EXP_ID, TEST = TEST)


