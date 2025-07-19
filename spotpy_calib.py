import glob
import logging
# from pathos.threading import ThreadPool as Pool
import multiprocessing as mp
import os
import shutil
import warnings
from os.path import join

import numpy as np
import pandas as pd
import tomlkit

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("esmvalcore")
logger.setLevel(logging.WARNING)
import json
import time
from datetime import datetime

import HydroErr as he
import matplotlib.pyplot as plt
import rioxarray as rioxr
import xarray as xr
from scipy import interpolate



from pathlib import Path

import numpy as np
from IPython.display import clear_output

from RunWflow_Julia import RunWflow_Julia, install_wflow, WflowJl, WflowJlForcing
from SnowClass import *
from SwissStations import *

# RUNDIR = os.getenv('EWC_RUNDIR', default = '/home/pwiersma/scratch/Scripts/Github/ewc/')
# os.chdir(RUNDIR)
# EWC_ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')

# EXP_ID = "Syn_912"
# BASIN = "Dischma"
# TEST = True
# CFG_DIR = f"{EWC_ROOTDIR}/experiments/config_files/{EXP_ID}_config.json"
# cfg = json.load(open(CFG_DIR))
# JOB_ID = 0
# YearlyCalib(cfg,)

from wflow_spot_setup import *


def SoilCalib(config, RUN_NAME):
    cfg = config
    cfg['SNOWMONTHS'] = None

    spotpy_setup = spot_setup(cfg, RUN_NAME,
                              calibration_purpose='Soilcalib')
    
    spotpy_output = join(cfg['SOILDIR'])

    sample_spotpy(cfg, spotpy_setup, spotpy_output, RUN_NAME)

def YearlyCalib(config, 
                YEAR,
                member,
                cluster_parset):
    cfg = config
    # print(cfg)
    print("Yearlycalib",YEAR)
    print(member,cluster_parset)
    RUN_NAME = f"{cfg['BASIN']}_{YEAR}_ms{member}_{cfg['EXP_ID']}"

    spotpy_setup = spot_setup(cfg, RUN_NAME,
                              calibration_purpose='Yearlycalib',
                              soil_parset= cluster_parset,
                               single_year = YEAR )
    
    spotpy_output = join(cfg['YEARLYDIR'],f"{YEAR}",f"{member}")
    Path(spotpy_output).mkdir(parents = True, exist_ok=True)
    sample_spotpy(cfg, spotpy_setup, spotpy_output, RUN_NAME)


def OB_calib(cfg,YEAR, member):
    print("OB_calib",YEAR)
    FOLDER = join(cfg['OUTDIR'],f"Soilcalib")
    FILE = join(FOLDER, f"{cfg['BASIN']}_ksoil{cfg['SOIL_K']}_posterior.csv")
    cluster_parsets = pd.read_csv(FILE,header=0, index_col=0)
    cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')
    cluster_parset = cluster_parsets_dic[member]


    RUN_NAME = f"{cfg['BASIN']}_{YEAR}_ms{member}_{cfg['EXP_ID']}_OB_calib"
    spotpy_setup = spot_setup(cfg, RUN_NAME,
                              calibration_purpose='OB_calib',
                              soil_parset= cluster_parset,
                              single_year = YEAR )
    
    spotpy_output = join(cfg['OUTDIR'],'OSHD',f"{YEAR}",f"{member}")
    Path(spotpy_output).mkdir(parents = True, exist_ok=True)
    sample_spotpy(cfg, spotpy_setup, spotpy_output, RUN_NAME)



def wrapper_function(cfg, m, pars, year):
    # Now a top-level function, can be pickled.
    return YearlyCalib(cfg, year, m, pars)

def YearlyCalib_joint(config,years):
    #load parsets from soilcalib 
    import subprocess

    cfg = config 

    if cfg['TEST'] == True:
        FOLDER = join(cfg['OUTDIR'],f"Soilcalib")
        FILE = join(FOLDER, f"{cfg['BASIN']}_ksoil{cfg['SOIL_K']}_posterior.csv")
        cluster_parsets = pd.read_csv(FILE,header=0, index_col=0)
        cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')
        for i,pardic in cluster_parsets_dic.items():
            YearlyCalib(cfg, cfg['START_YEAR'], i, cluster_parsets_dic[i])


    else:
        config_dir = join(cfg['EXPDIR'],"config_files",f"{cfg['EXP_ID']}_config.json")
        START_YEAR = cfg['START_YEAR']
        END_YEAR = cfg['END_YEAR']
        #make a string like [2001 2002 2003 ... 2010] betwee start and end year
        # years = " ".join([str(year) for year in years])
        print(years)
        
        # launch a bash script that runs YearlyCalib for each year
        code_dir = "/home/pwiersma/scratch/Scripts/Github/ewc/Yearlycalib.py"
        processes = []
        
        try:
            for year in years:
                command = ["python", code_dir, cfg['BASIN'], year, config_dir]
                process = subprocess.Popen(command)
                processes.append(process)

            # Wait for all subprocesses to finish
            for process in processes:
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)

        except subprocess.CalledProcessError as e:
            print(f"Process {e.cmd} failed with return code {e.returncode}. Terminating all processes.")
            for process in processes:
                if process.poll() is None:  # If the process is still running
                    process.terminate()
            for process in processes:
                process.wait()  # Ensure all processes have terminated
            raise
        # command = "for year in "+years+"; do python "+code_dir+" "+cfg['BASIN']+" $year "+config_dir+" & done"
        # subprocess.run(command, shell=True)

        # # Wait for all subprocesses to finish
        # subprocess.wait()

    
    # for year in range(cfg['START_YEAR'],cfg['END_YEAR']+1):
        #
   

    # FOLDER = join(cfg['OUTDIR'],f"Soilcalib")
    # FILE = join(FOLDER, f"{cfg['BASIN']}_{cfg['SOIL_K']}_clusters_posterior.csv")
    # cluster_parsets = pd.read_csv(FILE,header=0, index_col=0)
    # cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')

    # for m,pars in cluster_parsets_dic.items():      
    #     # wrapper = wrapper_function_factory(cfg,m, pars)
    #     # print(year,m,pars)
    #     # if threads is None:
    #     # with Pool() as pool:
    #     # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     #     # else:
    #     #     #     pool = Pool(nodes=threads)
    #     #     # Run parallel models
    #     #     Yearlyruns = pool.map(wrapper,
    #     #         np.arange(cfg['START_YEAR'],cfg['START_YEAR']+1))
    #     with mp.Pool(processes=mp.cpu_count()) as pool:
    #         Yearlyruns = pool.starmap(wrapper_function,
    #                                   [(cfg, m, pars, year) for year in np.arange(cfg['START_YEAR'], cfg['END_YEAR']+1)])

    #     # pool.close()
    #     # pool.join()




def sample_spotpy(config,
                  spotpy_setup, 
                  spotpy_output,
                  RUN_NAME):
    cfg = config
    print("Sampling with ",cfg['ALGO'])

    try: 
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print("MPI rank",rank)
        print("MPI size",size)
    except:
        print("MPI not available")

    algo = cfg['ALGO']
    if 'Soilcalib' in RUN_NAME: 
        N = cfg['N']    
    elif 'OB_calib' in RUN_NAME:
        N = 50
        algo = 'DDS'
    elif '_ms' in RUN_NAME:
        N = int(cfg['N']/2)

    try: 
        parallel = 'mpi' if cfg['PARALLEL'] else 'seq'
    except:
        parallel = 'mpi'
    # parallel = 'mpi'
    # N = 10


    if algo=='ROPE':
        sampler = spotpy.algorithms.rope(spotpy_setup, dbname = join(spotpy_output,RUN_NAME), dbformat = 'csv',
                                      parallel = parallel)
    elif algo =='MCMC':
        sampler = spotpy.algorithms.mcmc(spotpy_setup, dbname = join(spotpy_output,RUN_NAME), dbformat = 'csv')
    elif algo =='MC':
        sampler = spotpy.algorithms.mc(spotpy_setup, dbname = join(spotpy_output,RUN_NAME),dbformat = 'csv',
                                        parallel = parallel)
    elif algo =='LHS':
        sampler = spotpy.algorithms.lhs(spotpy_setup, dbname = join(spotpy_output,RUN_NAME),dbformat = 'csv',
                                        parallel = parallel)
    elif algo == 'SCEUA':
        sampler = spotpy.algorithms.sceua(spotpy_setup, dbname = join(spotpy_output,RUN_NAME), dbformat = 'csv',
                                        parallel = parallel)
    elif algo =='DDS':
        sampler = spotpy.algorithms.dds(spotpy_setup, dbname = join(spotpy_output,RUN_NAME), dbformat = 'csv',)
    elif algo =='DREAM':
        sampler = spotpy.algorithms.dream(spotpy_setup, dbname = join(spotpy_output,RUN_NAME), dbformat = 'csv',
                                        parallel = parallel)

    if algo =='ROPE':
        sampler.sample(N,
                        repetitions_first_run = int(cfg['PCT_FIRST_RUNS']*N), 
                        subsets = cfg['SUBSETS'], 
                        percentage_first_run = cfg['PCT_FIRST_RUNS'],
                        percentage_following_runs=0.05,
                        contains_list_params = False)
    elif algo =='SCEUA':
        sampler.sample(repetitions = N)#, ngs = 5,kstop = 50)
    else:
        sampler.sample(N)

