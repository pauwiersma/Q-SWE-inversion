import os
from os.path import join
from pathlib import Path


RUNDIR = os.getenv('EWC_RUNDIR', default = '/home/pwiersma/scratch/Scripts/Github/ewc/')
os.chdir(RUNDIR)

from spotpy_calib import SoilCalib
from spotpy_analysis import spotpy_analysis
from Synthetic_obs import Synthetic_obs
from Postruns import *
import argparse


#%% Yearly calib 
#check if there are soil_k spotpy files in each yearly folder 
# EXP_ID = "Syn_test4"
# BASIN = "Dischma"
# TEST = True


def soil_postproc(BASIN,EXP_ID,postruns_only = False,TEST = False):
    EWC_ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')
    CONFIG_DIR = join(join(EWC_ROOTDIR,"experiments",
        "config_files",f"{EXP_ID}_config.json"))
    config = json.load(open(CONFIG_DIR))
    config['BASIN'] = BASIN
    config['TEST'] = TEST
    SOILRUN_NAME = f"{config['BASIN']}_Soilcalib_{config['EXP_ID']}"

 #%% Spotpy soil calib
    if config['PERFECT_SOIL'] == False:
        clusterfile = join(config['SOILDIR'],f"{config['BASIN']}_ksoil{config['SOIL_K']}_{'posterior'}.csv")
        if os.path.isfile(clusterfile):
            print("Spotpy soil analysis done")
        else:
            SA  = spotpy_analysis(config, SOILRUN_NAME, 'Soilcalib').run()
    
    
    elif config['PERFECT_SOIL'] and config['SETTING'] == 'Synthetic':
        print("Skipping soilcalib")
        Path(config['SOILDIR']).mkdir(parents = True, exist_ok=True)
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
    soil_postproc(BASIN,EXP_ID, TEST = TEST)