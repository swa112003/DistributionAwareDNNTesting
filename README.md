# Distribution-Aware Testing of Neural Networks Using Generative Models

Releasing Python code for the paper ["Distribution-Aware Testing of Neural Networks Using Generative Models"]()

## Introduction
This directory contains the implementation of the test generation framework proposed in our work. 
The framework generates valid test inputs for testing DNNs. The code is developed 
using DeepXplore implementation as a baseline.

## Citation


## Dependencies
Requires Linux Platform with `Python 3.6.2` installed.

Create virtualenv using below command

`python -m venv ./venv`

Activate virtual environment

`source ./venv/bin/activate`

requirements.txt file contains the python packages required for running the code.
Packages can be installed using below command

`pip install -r requirements.txt`

## Files
- `MNIST` Directory contains scripts for MNIST dataset
- `SVHN` Directory contains scripts for SVHN dataset
- `SVHN/dataset` Directory should contain SVHN dataset files

Note: 
- The trained model for SVHN/ModelE is not included due to file size limitations. It can be trained using the provided ModelE.py file.
- For SVHN tests, copy .mat files from below location to `SVHN\dataset` directory.
	
	http://ufldl.stanford.edu/housenumbers

## How to use 

### Method1
Below procedure can be used to run tests for a subset of seeds used in the evaluation section of the paper.
```
For running MNIST scripts for individual models:
cd MNIST 
./run_MNIST.sh <1/2/3/4>
#1 - MNI-1, 2 - MNI-2, 3 - MNI-3, and 4 - MNI4

For running SVHN scripts for individual models:
cd SVHN
./run_SVHN.sh <1/2/3/4>
#1 - SVH-1, 2 - SVH-2, 3 - SVH-3, and 4 - SVH-4
```
#### Output
The script will output the number of test inputs generated, test coverage and time taken for generating the test inputs to stdout.
It will save the generated tests in generated_inputs_Model# and baseline_generated_inputs_Model# directories.

### Method2
Run below commands to reproduce the results in the paper.

```
To run all MNIST tests used in the paper (MNI-1, MNI-2, MNI-3 and MNI-4):
cd MNIST
./run_MNIST_all.sh

To run all SVHN tests used in the paper (SVH-1, SVH-2, SVH-3 and SVH-4):
cd SVHN
./run_SVHN_all.sh
```

#### Output
The script will output the number of test inputs generated, test coverage and time taken for generating the test inputs to `output.txt` file.
It will save the generated tests in generated_inputs_Model# and baseline_generated_inputs_Model# directories.

### Method3
Run `dist_gen_diff.py` in each directory to run the scripts with custom parameters.

Descriptions for different options are provided below:

```
python dist_gen_diff.py 
	[constraint=light/occl/blackout] 
	[weight of DNN differential behavior] 
	[weight of neuron activation]
	[weight of VAE density]
	[No of seeds] 
	[maximum iterations] 
	[neuron coverage threshold] 	
	[model under test=0/1/2/3]
```
#### Example						

```
cd MNIST
python dist_gen_diff.py occl 3 .5 .1 .1 200 30 .25 --target_model=2
```
```
cd SVHN
python dist_gen_diff.py occl 1 .5 5 1 200 30 .25 --target_model=0
```

For running baseline, set `weight of VAE density` to 0 in the above commands.


#### Output
The script will output the number of test inputs generated, test coverage and time taken for generating the test inputs to stdout.
It will save the generated tests in generated_inputs_Model# and baseline_generated_inputs_Model# directories.
