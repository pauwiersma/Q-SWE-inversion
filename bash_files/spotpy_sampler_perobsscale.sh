
#!/bin/bash

# Loop through the years and launch a Python job for each one
#for obsscale in -0.3 0.0 0.1 0.2 0.3 0.4 0.5; do
for obsscale in 1.0 2.0 4.0 8.0; do
    # Pass the catchment name as an argument to your Python script
    python spotpy_SR.py $obsscale &
done

# Wait for all background jobs to finish
wait

echo "All jobs have completed."

