# Code accompanying "Can streamflow constrain snow mass reconstruction? Lessons from two synthetic experiments"

This repository contains scientific code and is not actively maintained. The compilation of the correct hydrological model version and python packages is not straightforward, if you're interested in reproducing the experiments using the code presented here, please contact me at pauwiersma@outlook.com. 
If you're interested in reproducing the experiments with your own snow and runoff models, we refer to the Zenodo repository for the necessary data: 10.5281/zenodo.16146617. 

## Overview

This project implements a streamflow-constrained SWE reconstruction framework that:
- Prepares all necessary input data and loads the wflow_sbm hydrological model for a catchment of choice
- Reads a config file containing all necessary model parameters to initialize the model
- Performs repeated model executions with different parameter sets sampled using different algorithms from the SPOTPY package
- Compares against observed streamflow and SWE observations, analyzes results and produces plots
- Supports both local and cluster-based computations

## Main Scripts

### Preprocessing and Setup

- **`preproc.py`** - Preprocessing module that sets up experiment configurations, creates directories, and prepares parameter ranges for calibration
- **`generate_forcing.py`** - Generates meteorological forcing data for the hydrological model
- **`wflow_spot_setup.py`** - Sets up wflow model configurations for spotpy calibration framework

### Model Execution

- **`RunWflow_Julia.py`** - Executes the wflow hydrological model using Julia backend
- **`LOA.py`** - Large-scale analysis class for processing and analyzing model outputs

### Computation Scripts

- **`yearly_compute.py`** - Performs yearly computations for model calibration
- **`yearly_postproc.py`** - Post-processes yearly computation results
- **`soil_compute.py`** - Computes soil-related model parameters
- **`soil_postproc.py`** - Post-processes soil computation results

### Calibration and Analysis

- **`spotpy_calib.py`** - Implements calibration using the spotpy framework
- **`spotpy_analysis.py`** - Analyzes calibration results from spotpy
- **`Yearlycalib.py`** - Handles yearly calibration procedures

### Metrics and Evaluation

- **`calc_swe_q_metrics.py`** - Calculates comprehensive SWE and discharge metrics
- **`swe_metrics.py`** - Contains SWE-specific metric calculations
- **`q_metrics.py`** - Contains discharge-specific metric calculations
- **`Evaluation.py`** - Comprehensive evaluation framework for model performance

### Data Processing

- **`Synthetic_obs.py`** - Generates synthetic observations for testing
- **`Postruns.py`** - Handles post-run analysis and processing
- **`SwissStations.py`** - Processes Swiss meteorological station data
- **`SnowClass.py`** - Snow classification and processing utilities

## Utility Scripts (`utils/` directory)

- **`dump.py`** - Data dumping and loading utilities
- **`wflow_baseflow.py`** - Baseflow separation utilities
- **`hydrobricks_catchment.py`** - Catchment processing using hydrobricks
- **`spatial_stuff.py`** - Spatial data processing utilities
- **`MeteoSwiss_to_ERA5.py`** - Converts MeteoSwiss data to ERA5 format
- **`MeteoSwiss2wflow.py`** - Converts MeteoSwiss data for wflow input
- **`MeteoSwiss2wflowJulia.py`** - Converts MeteoSwiss data for wflow Julia
- **`generate_synthetic_obs.py`** - Generates synthetic observations
- **`kmeans_clustering.py`** - Implements k-means clustering for data analysis
- **`cluster_hydrograph_ranges.py`** - Clusters hydrograph ranges
- **`delete_output_folders.py`** - Utility to clean output directories
- **`zip_configfiles.py`** - Archives configuration files

## SLURM Scripts

The repository includes several SLURM scripts for cluster execution:
- **`master_slurm.run`** - Master SLURM script for orchestrating jobs
- **`SLURM_*.run`** - Various SLURM scripts for different computation stages

## Configuration Files

- **`ewatercycle_config.yaml`** - eWaterCycle configuration
- **`ewc_wflowjl_infra.yml`** - wflow Julia infrastructure configuration
- **`ewc_wflowjl_infra_nojupyter.yml`** - wflow Julia configuration without Jupyter

## Bash Scripts (`bash_files/` directory)

Contains various bash scripts for:
- Yearly calibration
- spotpy settings and execution
- Parallel processing configurations
  
## Dependencies

The codebase requires various Python packages including:
- numpy, pandas, xarray
- matplotlib, seaborn
- rasterio, geopandas
- spotpy, hydrosignatures
- wflow (Julia backend)
- Various hydrological modeling libraries


- This repository is designed for research purposes in hydrological modeling
- The code supports both synthetic and real-world experiments
- Cluster execution is supported through SLURM job scheduling
- The framework is particularly focused on snow-dominated catchments 
