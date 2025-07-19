
#!/bin/bash

# Loop through the years and launch a Python job for each one
for basin in Verzasca Jonschwil Landquart Landwasser Rom Mogelsberg Dischma Ova_da_Cluozza Sitter Werthenstein Sense Ilfis Eggiwil Chamuerabach Veveyse Minster Ova_dal_Fuorn Alp Biber Riale_di_Calneggia Chli_Schliere Allenbach; do
    # Pass the catchment name as an argument to your Python script
    python wflow_julia_postruns_singleR.py $basin &
done

# Wait for all background jobs to finish
wait

echo "All jobs have completed."

