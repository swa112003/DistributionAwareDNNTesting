#Usage run_MNIST_all.sh

#!/bin/bash

echo "******Testing for MNI-1******" > output.txt;
#Baseline + VAE
python dist_gen_diff.py occl 3 .5 .1 .1 200 30 .25 --target_model=0 >> output.txt;
#Baseline
python dist_gen_diff.py occl 3 .5 0 .1 200 30 .25 --target_model=0 >> output.txt;

echo >> output.txt;
echo "******Testing for MNI-2******" >> output.txt;
#Baseline + VAE
python dist_gen_diff.py occl 3 .5 .1 .1 200 30 .25 --target_model=1 >> output.txt;
#Baseline
python dist_gen_diff.py occl 3 .5 0 .1 200 30 .25 --target_model=1  >> output.txt;

echo >> output.txt;
echo "******Testing for MNI-3******" >> output.txt;
#Baseline + VAE
python dist_gen_diff.py occl 3 .5 .1 .1 200 30 .25 --target_model=2 >> output.txt;
#Baseline
python dist_gen_diff.py occl 3 .5 0 .1 200 30 .25 --target_model=2 >> output.txt;

echo >> output.txt;
echo "******Testing for MNI-4******" >> output.txt;
#Baseline + VAE
python dist_gen_diff.py occl 3 .5 .1 .1 200 30 .25 --target_model=3 >> output.txt;
#Baseline
python dist_gen_diff.py occl 3 .5 0 .1 200 30 .25 --target_model=3 >> output.txt;
