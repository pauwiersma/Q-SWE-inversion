#!/bin/bash
# List of catchment names

# catchments=("Mogelsberg" "Dischma" "Ova_da_Cluozza" "Sitter" "Werthenstein" "Sense" "Ilfis" "Eggiwil" "Chamuerabach" "Veveyse" "Minster" "Ova_dal_Fuorn" "Alp" "Biber" "Muota" "Riale_di_Calneggia" "Chli_Schliere" "Allenbach" "Jonschwil" "Rom" "Verzasca" "Landwasser" "Landquart")
# catchments=("Dischma" "Ova_da_Cluozza" "Ova_dal_Fuorn" "Alp" "Biber" "Riale_di_Calneggia" "Chli_Schliere" "Allenbach" "Landwasser" "Rom" "Verzasca")
# catchments = ("Riale_di_Calneggia")
catchment="Riale_di_Calneggia"
settings=("Real" "Synthetic" "Soilonly" "OSHD_EB")

# catchments=("Jonschwil" "Landquart" "Landwasser" "Rom" "Verzasca")
# Loop through the catchments and launch a Python job for each one
for setting in "${settings[@]}"; do
    # Pass the catchment name as an argument to your Python script
    python spotpy_SR.py "$catchment" "$setting" &
done

# Wait for all background jobs to finish
wait

echo "All jobs have completed."
