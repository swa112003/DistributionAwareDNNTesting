#Usage run_SVHN.sh <model=1,2,3,4>
#Example run_SVHN.sh 1

#!/bin/bash

model=$(( $1 - 1 ))

echo "Running tests for SVH-"$1;
#Baseline + VAE
python dist_gen_diff.py occl 1 .5 5 1 50 30 .25 --target_model=$model;

#Baseline
python dist_gen_diff.py occl 1 .5 0 1 50 30 .25 --target_model=$model;
