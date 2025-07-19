

#%%
"""
Module for running yearly member calibration.
"""

import json
import logging
import os
import sys
import warnings
from os.path import join
from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
from spotpy_calib import spot_setup, sample_spotpy

warnings.filterwarnings("ignore", category=UserWarning)

EWC_RUNDIR = os.getenv('EWC_RUNDIR', default = '/home/pwiersma/scratch/Scripts/Github/ewc/')
os.chdir(EWC_RUNDIR)
EWC_ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')

#test
BASIN = 'Dischma'
CFG_DIR = join(join(EWC_ROOTDIR,"experiments/config_files","Syn_test4_config.json"))
TEST = True

def soil_compute(BASIN , CFG_DIR, TEST = False ):
    """
    Perform yearly member calibration for a given basin.
    Parameters:
    BASIN (str): The name of the basin.
    CFG_DIR (str): The directory path to the configuration file.
    JOB_ID (int): The job identifier.
    Returns:
    None
    This function reads the configuration file and the posterior CSV file for the given basin and soil parameters.
    It calculates the single year and member based on the JOB_ID and configuration settings.
    It then sets up the SPOTPY calibration for the specified year and member, creates the necessary output directory,
    and runs the SPOTPY sampling process.
    """
    time0 = time.time()
    print('pathexists',Path("/work/FAC/FGSE/IDYST/gmariet1/gaia/pwiersma/ewatercycle/outputs/Syn_2510test6/Yearlycalib").exists())
    cfg = json.load(open(CFG_DIR))
    cfg['BASIN'] = BASIN
    cfg['TEST'] = TEST

    cfg['SNOWMONTHS'] = None

    try: 
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except:
        rank = 0

    if cfg['PERFECT_SOIL'] == False:
        

        SOILRUN_NAME = f"{cfg['BASIN']}_Soilcalib_{cfg['EXP_ID']}"

        if os.path.exists(cfg['SOILDIR']) and (os.path.isfile(join(cfg['SOILDIR'],f"{SOILRUN_NAME}.csv"))) and not "work" in cfg['ROOTDIR']:
        # if os.path.isfile(SOILFILE):
            print("Soil calib done")
        else:
            Path(cfg['SOILDIR']).mkdir(parents = True, exist_ok=True)
            # SoilCalib(cfg,SOILRUN_NAME)

            spotpy_setup = spot_setup(cfg, SOILRUN_NAME,
                                    calibration_purpose='Soilcalib')
            
            spotpy_output = join(cfg['SOILDIR'])

            sample_spotpy(cfg, spotpy_setup, spotpy_output, SOILRUN_NAME)
            print("Soil calib done")
            print("Time elapsed",time.time()-time0, "s")
    else:
        print("Perfect soil; skipping soil calib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('BASIN', type=str, help='The basin identifier')
    parser.add_argument('CFG_DIR', type=str, help='The configuration file directory')
    parser.add_argument('--test', dest = 'TEST', type=str, choices=['true', 'false', '1', '0'], default='false', help='Run in test mode')

    args = parser.parse_args()

    BASIN = args.BASIN
    CFG_DIR = args.CFG_DIR
    if args.TEST in ['true', '1']:
        TEST = True
    else:
        TEST = False
    soil_compute(BASIN, CFG_DIR, TEST = TEST)
