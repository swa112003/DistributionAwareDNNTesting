#Usage run_SVHN_all.sh


#!/bin/bash

echo "Running tests for SVH-1" > output.txt;
#Baseline + VAE
python dist_gen_diff.py occl 1 .5 5 1 200 30 .25 --target_model=0 >> output.txt;
#Baseline
python dist_gen_diff.py occl 1 .5 0 1 200 30 .25 --target_model=0 >> output.txt;

echo "Running tests for SVH-2" >> output.txt;
#Baseline + VAE
python dist_gen_diff.py occl 1 .5 5 1 200 30 .25 --target_model=1 >> output.txt;
#Baseline
python dist_gen_diff.py occl 1 .5 0 1 200 30 .25 --target_model=1 >> output.txt;

echo "Running tests for SVH-3" >> output.txt;
#Baseline + VAE
python dist_gen_diff.py occl 1 .5 5 1 200 30 .25 --target_model=2 >> output.txt;
#Baseline
python dist_gen_diff.py occl 1 .5 0 1 200 30 .25 --target_model=2 >> output.txt;

echo "Running tests for SVH-4" >> output.txt;
#Baseline + VAE
python dist_gen_diff.py occl 1 .5 5 1 200 30 .25 --target_model=3 >> output.txt;
#Baseline
python dist_gen_diff.py occl 1 .5 0 1 200 30 .25 --target_model=3 >> output.txt;
