#Usage run_MNIST.sh <model=1,2,3,4>
#Example run_MNIST.sh 1

#!/bin/bash

model=$(( $1 - 1 ))

echo "Running tests for MNI-"$1;
#Baseline + VAE
python dist_gen_diff.py occl 3 .5 .1 .1 50 20 .25 --target_model=$model;

#Baseline
python dist_gen_diff.py occl 3 .5 0 .1 50 20 .25 --target_model=$model;
