#!/bin/bash
# List of catchment names
#"Werthenstein" "Muota"
catchments=("Mogelsberg" "Dischma" "Ova_da_Cluozza" "Sitter"  "Eggiwil" "Chamuerabach" "Minster" "Ova_dal_Fuorn" "Alp" "Chli_Schliere" "Allenbach" "Rom" "Verzasca" "Landwasser")
# catchments=("Dischma" "Ova_da_Cluozza" "Ova_dal_Fuorn" "Alp" "Biber" "Riale_di_Calneggia" "Chli_Schliere" "Allenbach" "Landwasser" "Rom" "Verzasca")
# catchments = ("Riale_di_Calneggia")
settings=("Real" "OSHD_TI" "Soilonly")
#"Soilonly")
#"Soilonly" "OSHD_EB")

# catchments=("Jonschwil" "Landquart" "Landwasser" "Rom" "Verzasca")
# Loop through the catchments and launch a Python job for each one
for setting in "${settings[@]}"; do
    # Loop through the catchments and launch a Python job for each one
    for catchment in "${catchments[@]}"; do
        # Pass the catchment name as an argument to your Python script
        python wflow_julia_postruns_singleR.py "$catchment" "$setting" &
    done

    # Wait for all background jobs to finish before moving on to the next setting
    wait
done

echo "All jobs have completed."
