# Focused-view CTA for selective visualization of stroke related arteries
This repository contains the experimental code and analysis scripts used for the study "Focused view CT angiography for selective visualization of stroke related arteries: technical feasibility".
```
@article{roest2023focused,
  title={Focused view CT angiography for selective visualization of stroke related arteries: technical feasibility},
  author={Roest, Christian and Kloet, Reina W and Lamers, Maria J and Yakar, Derya and Kwee, Thomas C},
  journal={European Radiology},
  pages={1--10},
  year={2023},
  publisher={Springer}
}
}
```

### Requirements
Deep learning was performed in Python version 3.7.4.
The following requirements apply:
```
tensorflow>=2.2.0
scikit-learn==1.2.2
SimpleITK>=2.0.2
numpy>=1.21.3
tqdm
```

For our experiments we used a python virtual environment (venv) to manage installed packages.
This can be set up with the following commands: 
```
### Create the virtual environment
python -m venv env

### Load the environment (choose which applies)
### Linux
source env/bin/activate

### Windows
env/Scripts/activate.ps1

### Install packages to the enviroment
pip install tensorflow==2.2.0
pip install scikit-learn==1.2.2
pip install SimpleITK==2.0.2
pip install numpy==1.21.3
pip install tqdm
```

After the environment has been set up once, you can load it in the future by running:
```
### Linux
source env/bin/activate

### Windows
env/Scripts/activate.ps1
```

### 1. Hyperparameter optimization
We used Optuna to optimize the hyperparameters for each radiomics model.
Optuna searches the search space to optimize an objective function, both of which are defined in `optimize.py`.
To start a hyperparameter search for a set of radiomics features extracted in Step 1, run the following command:
```
python optimization.py 
```
This will create a database containing Optuna results in `ct_stroke_opt.db`.

Optuna also supports distributed optimization, for which a central database is used to store objective values for each candidate set of hyperparameters.
An example SLURM job-script that was used to run a distributed hyperparameter search across multiple CPU jobs is shown in `opt.job`.
To run this script on your HPC cluster, modules / resources / partitions needs to be adjusted to your cluster's configuration. 

### 2. Train U-Net ensemble using optimal hyperparameters
To use the optimal settings derived in the previous step and train a U-Net with them, run `train.py`, or submit `train.job` to your SLURM scheduler.
This will automatically extract the optimal settings from the SQLite database.

### 3. Inference
To predict on the held-out test set, use `predict.py`. This will segment provided test-scans, and generate segmentation maps as well as focused-view scans by masking unsegmented regions.
