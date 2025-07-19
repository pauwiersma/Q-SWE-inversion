import os
from os.path import join

ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')
RUNDIR = os.getenv('EWC_RUNDIR', default = '/home/pwiersma/scratch/Scripts/Github/ewc/')
os.chdir(RUNDIR)
from yearly_postproc import yearly_postproc
from yearly_compute import yearly_compute
from preproc import preproc
from soil_compute import soil_compute
from soil_postproc import soil_postproc
import concurrent.futures
import subprocess



from spotpy_calib import SoilCalib
from spotpy_analysis import spotpy_analysis
from Synthetic_obs import Synthetic_obs
from Postruns import *
from Evaluation import *

EXP_ID = "Syn_1311e"
BASIN = "Riale_di_Calneggia"
TEST = True

#preproc
cfg = preproc(BASIN,EXP_ID,TEST = TEST)

NJOBS = cfg['SOIL_K'] * (cfg['END_YEAR'] - cfg['START_YEAR'] + 1)
JOB_ARRAY = np.arange(0, NJOBS)
CFG_DIR = join(ROOTDIR, "experiments", "config_files", f"{EXP_ID}_config.json")

soil_compute(BASIN,CFG_DIR,TEST = TEST)
soil_postproc(BASIN,EXP_ID,TEST = TEST)

def run_command(basin, cfg_dir, job_id):
    command = ["python3", "yearly_compute.py", basin, cfg_dir, str(job_id)]
    subprocess.run(command)

if not "/home/pwiersma" in ROOTDIR:
    # running on cluster with slurm
    for JOB_ID in JOB_ARRAY:
        run_command(BASIN, CFG_DIR, JOB_ID)
else:
    # running on local machine
    def run_batch(IDs):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_command, BASIN, CFG_DIR, JOB_ID) for JOB_ID in IDs]
            concurrent.futures.wait(futures)

    # Run jobs in batches NYEARS
    for k in range(0, cfg['SOIL_K']):
        IDs = np.arange(k, NJOBS, cfg['SOIL_K'])
        print(IDs, " of ", NJOBS)
        run_batch(IDs)


#compute
# NJOBS = cfg['SOIL_K'] * (cfg['END_YEAR'] - cfg['START_YEAR']+1)
# JOB_ARRAY = np.arange(0, NJOBS)
# CFG_DIR = join(ROOTDIR,"experiments","config_files",f"{EXP_ID}_config.json")
# if not "/home/pwiersma" in ROOTDIR:
#     #running on cluster with slurm
#     for JOB_ID in JOB_ARRAY:
#         compute(BASIN , CFG_DIR, JOB_ID)
# else:
#     #running on local machine
#     def run_batch(IDs):
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             futures = [executor.submit(compute, BASIN, CFG_DIR, JOB_ID) for JOB_ID in IDs]
#             concurrent.futures.wait(futures)

#     # Run jobs in batches NYEARS
#     for k in range(0, cfg['SOIL_K']):
#         IDs = np.arange(k, NJOBS, cfg['SOIL_K'])
#         print(IDs, " of ", NJOBS)
#         run_batch(IDs)

#postproc
yearly_postproc(BASIN,EXP_ID, TEST = TEST)
