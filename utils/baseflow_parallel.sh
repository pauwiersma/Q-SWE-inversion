#!/bin/bash
basins="Mogelsberg Dischma Ova_da_Cluozza Sitter Werthenstein Sense Ilfis Eggiwil Chamuerabach Veveyse Minster Ova_dal_Fuorn Alp Biber Riale_di_Calneggia Chli_Schliere Allenbach Jonschwil Landwasser Landquart Rom Verzasca Muota"

for basin in $basins; do
    echo "Processing basin: $basin"
    python wflow_baseflow.py $basin &
    # Add your processing commands here
done

# Wait for all background jobs to finish
