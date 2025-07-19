#!/bin/bash

# Loop through the years and launch a Python job for each one
for year in {2005..2015}; do
    # Pass the catchment name as an argument to your Python script
    python spotpy_yearly_sampler.py Jonschwil $year &
done

# Wait for all background jobs to finish
wait

echo "All jobs have completed."
