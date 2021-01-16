# Distribution-Aware Testing of Neural Networks Using Generative Models
This repository contains the artifact for testing Deep Neural Networks(DNNs). It is implemented in Python and includes the code and models required for replicating the studies in the paper ["Distribution-Aware Testing of Neural Networks Using Generative Models"]().

## Introduction
The artifact implements the test generation framework proposed in our work. The framework generates valid test inputs for testing DNNs trained on MNIST [2] and SVHN [3] datasets. The code is developed using DeepXplore [1] implementation as a baseline. It uses Keras library and we tested it with Tensorflow backend on GPUs.

## Environment Setup
Requires Linux Platform with `Python 3.6.2` and `virtualenv` installed.

`requirements.txt` file contains the python packages required for running the code.
Follow below steps for installing the packages:
- Create virtual environment

	`python -m venv ./venv`
- Activate virtual environment

	`source ./venv/bin/activate`
- Install packages using pip

	`pip install -r requirements.txt`

## Files
- `MNIST` Directory contains scripts for MNIST dataset
- `SVHN` Directory contains scripts for SVHN dataset
- `SVHN/dataset` Placeholder for SVHN dataset files

##### Note: 
- The trained model for SVHN/ModelE is not included due to file size limitations. It can be trained using the provided `ModelE.py` file.
- For SVHN tests, copy `.mat` files from below location to `SVHN/dataset` directory.
	http://ufldl.stanford.edu/housenumbers

## How To Use 
The tool can be run using the provided shell scripts or by invoking `dist_gen_diff.py` script.

### Verify Tool Setup
`run_MNIST.sh` can be used to run tests for a small set of seeds to verify that the tool is setup properly. Each test will the run the tool for VAE+baseline and baseline and should take less than 30 mins to run. The execution time will vary depending on the platform used.
##### Procedure
For running MNIST tests:

N in below command is between 1 and 4 and designates MNIST model **MNI-N** from the paper.
``` 
cd MNIST 
./run_MNIST.sh N
```
For running SVHN tests:

N in below command is between 1 and 4 and designates SVHN model **SVH-N** from the paper.
```
cd SVHN
./run_SVHN.sh N
```
##### Output
The script will print out the number of tests generated, neuron coverage and average time and the overall time taken for generating the tests to stdout. It will save the generated tests in `generated_inputs_Model#` and `baseline_generated_inputs_Model#` directories. When the number of tests generated is zero, it will print out NA for average test generation time. The test generation time will vary depending on the execution platform.

Below is a sample output for MNI-1 model:
```
Running tests for MNI-1
***** Result of VAE+Baseline test:
No of test inputs generated: 5
Cumulative coverage for tests: 0.346
Avg. test generation time: 46.24 s
Total time: 231.18 s
***** Result of baseline test:
No of test inputs generated: 0
Cumulative coverage for tests: 0.0
Avg. test generation time: NA s
Total time: 407.65 s
```

### Run Tests Used In The Paper
`run_MNIST_all.sh` and `run_SVHN_all.sh` shell scripts are provided to run all the test generation tests used in the paper. Run below commands to reproduce the results in the paper. Running all the tests should take around 10 hrs on GPUs. The execution time will vary depending on the platform used.
#### Procedure
To run all MNIST tests used in the paper (MNI-1, MNI-2, MNI-3 and MNI-4):
```
cd MNIST
./run_MNIST_all.sh
```
To run all SVHN tests used in the paper (SVH-1, SVH-2, SVH-3 and SVH-4):
```
cd SVHN
./run_SVHN_all.sh
```
##### Output
The script will save the output for all the models in `output.txt` file. It will contain output similar to the one mentioned above for each of the models.

### Run The Tool With Custom Parameters
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
#### Examples						
```
cd MNIST
python dist_gen_diff.py occl 3 .5 .1 .1 200 30 .25 --target_model=2
```
```
cd SVHN
python dist_gen_diff.py occl 1 .5 5 1 200 30 .25 --target_model=0
```
Set `weight of VAE density` to 0 in the above commands for running baseline tests.

## Using The Tool For New Models
The implementation can easily be extended to test new models. To use the tool for new models, add keras model files and their respective trained H5 files to the MNIST or SVHN directory. Make appropriate changes in `dist_gen_diff.py` for it to load these new model files. 

## References
1. Pei, Kexin, et al. "Deepxplore: Automated whitebox testing of deep learning systems." proceedings of the 26th Symposium on Operating Systems Principles. 2017.

	https://github.com/peikexin9/deepxplore
	
2. LeCun, Yann. "The MNIST database of handwritten digits." http://yann.lecun.com/exdb/mnist/ (1998).
3. Netzer, Yuval, et al. "Reading digits in natural images with unsupervised feature learning." (2011).
