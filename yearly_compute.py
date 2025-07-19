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

EWC_ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')
EWC_RUNDIR = os.getenv('EWC_RUNDIR', default = '/home/pwiersma/scratch/Scripts/Github/ewc/')
os.chdir(EWC_RUNDIR)

EXP_ID = "Syn_912"
BASIN = "Dischma"
TEST = True
CFG_DIR = f"{EWC_ROOTDIR}/experiments/config_files/{EXP_ID}_config.json"
JOB_ID = 0

def yearly_compute(BASIN , CFG_DIR, JOB_ID, TEST = False ):
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
    JOB_ID = int(JOB_ID)
    FOLDER = join(cfg['OUTDIR'],f"Soilcalib")
    FILE = join(FOLDER, f"{cfg['BASIN']}_ksoil{cfg['SOIL_K']}_posterior.csv")
    cluster_parsets = pd.read_csv(FILE,header=0, index_col=0)
    cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')
    single_year = int(np.floor(JOB_ID/cfg['SOIL_K']))+cfg['START_YEAR']
    member = JOB_ID%cfg['SOIL_K']
    cluster_parset = cluster_parsets_dic[member]

    print("Yearlycalib",single_year, member)
    RUN_NAME = f"{cfg['BASIN']}_{single_year}_ms{member}_{cfg['EXP_ID']}"

    spotpy_setup = spot_setup(cfg, RUN_NAME,
                              calibration_purpose='Yearlycalib',
                              soil_parset= cluster_parset,
                               single_year = single_year )
    
    spotpy_output = join(cfg['YEARLYDIR'],f"{single_year}",f"{member}")
    try: 
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except:
        rank = 0
        
    if Path(spotpy_output).exists() and rank == 0 and not "work" in cfg['ROOTDIR']:
        print("Output directory already exists. Skipping.")
        return
    else:
        Path(spotpy_output).mkdir(parents = True, exist_ok=True)
        sample_spotpy(cfg, spotpy_setup, spotpy_output, RUN_NAME)
        print("Computations done for ", single_year, member)
        print("Time taken: ", time.time()-time0, " seconds")


#testing
# EXP_ID = "Syn_test21"
# BASIN = "Riale_di_Calneggia"
# CFG_DIR = f"/home/pwiersma/scratch/Data/ewatercycle/experiments/config_files/{EXP_ID}_config.json"
# JOB_ID = 0
# TEST = True 
# compute(BASIN, CFG_DIR, JOB_ID, TEST = TEST)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('BASIN', type=str, help='The basin identifier')
    parser.add_argument('CFG_DIR', type=str, help='The configuration file directory')
    parser.add_argument('JOB_ID', type=str, help='The job ID')
    parser.add_argument('--test', dest = 'TEST', type=str, choices=['true', 'false', '1', '0'], default='false', help='Run in test mode')

    args = parser.parse_args()

    BASIN = args.BASIN
    CFG_DIR = args.CFG_DIR
    JOB_ID = args.JOB_ID
    if args.TEST in ['true', '1']:
        TEST = True
    else:
        TEST = False
        
    yearly_compute(BASIN, CFG_DIR, JOB_ID, TEST = TEST)