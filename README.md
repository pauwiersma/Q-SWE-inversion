# Q-SWE Inversion

This repository contains scripts for hydrological modeling and analysis, specifically focused on Snow Water Equivalent (SWE) and discharge (Q) inversion using the wflow hydrological model. The codebase is designed for calibration, evaluation, and analysis of hydrological models with a particular emphasis on snow processes.

## Overview

This project implements a comprehensive hydrological modeling framework that:
- Calibrates hydrological models using both discharge and SWE observations
- Performs synthetic experiments to test model performance
- Analyzes model outputs using various hydrological signatures and metrics
- Supports both local and cluster-based computations

## Main Scripts

### Core Execution Scripts

- **`main.py`** - Main execution script that orchestrates the entire workflow including preprocessing, soil computation, yearly computation, and postprocessing
- **`main_old.py`** - Previous version of the main execution script (deprecated)

### Preprocessing and Setup

- **`preproc.py`** - Preprocessing module that sets up experiment configurations, creates directories, and prepares parameter ranges for calibration
- **`generate_forcing.py`** - Generates meteorological forcing data for the hydrological model
- **`wflow_spot_setup.py`** - Sets up wflow model configurations for spotpy calibration framework

### Model Execution

- **`RunWflow_Julia.py`** - Executes the wflow hydrological model using Julia backend
- **`wflow_julia_calibration.py`** - Handles wflow model calibration using Julia
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

### Graphics and Visualization

- **`Graphics.py`** - Visualization utilities for plotting results

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

## Key Features

1. **Multi-objective Calibration**: Supports calibration on both discharge and SWE observations
2. **Synthetic Experiments**: Framework for testing with synthetic data
3. **Parallel Processing**: Supports both local and cluster-based parallel computation
4. **Comprehensive Metrics**: Implements various hydrological signatures and performance metrics
5. **Flexible Configuration**: JSON-based configuration system for easy experiment setup
6. **Multiple Algorithms**: Supports various calibration algorithms (LHS, ROPE, MC, DDS)

## Usage

1. Set up environment variables:
   - `EWC_ROOTDIR`: Root directory for data
   - `EWC_RUNDIR`: Runtime directory

2. Configure experiment parameters in `preproc.py`

3. Run the main execution script:
   ```bash
   python main.py
   ```

## Dependencies

The codebase requires various Python packages including:
- numpy, pandas, xarray
- matplotlib, seaborn
- rasterio, geopandas
- spotpy, hydrosignatures
- wflow (Julia backend)
- Various hydrological modeling libraries

## Notes

- This repository is designed for research purposes in hydrological modeling
- The code supports both synthetic and real-world experiments
- Cluster execution is supported through SLURM job scheduling
- The framework is particularly focused on snow-dominated catchments 