# Unified Model Shift and Model Bias Policy Optimization

Code to reproduce the experiments in [How to Fine-tune the Model: Unified Model Shift and
Model Bias Policy Optimization].


## Installation
1. Install [MuJoCo 1.50](https://www.roboti.us/index.html) at `~/.mujoco/mjpro150` and copy your license key to `~/.mujoco/mjkey.txt`
2. Clone `usb`
```
git clone https://github.com/betray12138/Unified-Model-Shift-and-Model-Bias-Policy-Optimization.git
```
3. Create a conda environment and install usb
```
cd Unified-Model-Shift-and-Model-Bias-Policy-Optimization
conda env create -f environment/gpu-env.yml
conda activate usb
pip install -e viskit
pip install -e .
```

## Usage
Configuration files can be found in [`examples/config/`](examples/config).

```
mbpo run_local examples.development --config=examples.config.halfcheetah.0 --gpus=1 --trial-gpus=1
```

Currently only running locally is supported.

#### Logging

This codebase contains [viskit](https://github.com/vitchyr/viskit) as a submodule. You can view saved runs with:
```
viskit ~/ray_mbpo --port 6008
```
assuming you used the default [`log_dir`](examples/config/halfcheetah/0.py#L7).

#### Hyperparameters

The rollout length schedule is defined by a length-4 list in a [config file](examples/config/halfcheetah/0.py#L31). The format is `[start_epoch, end_epoch, start_length, end_length]`, so the following:
```
'rollout_schedule': [20, 100, 1, 5] 
```
corresponds to a model rollout length linearly increasing from 1 to 5 over epochs 20 to 100. 

If you want to speed up training in terms of wall clock time (but possibly make the runs less sample-efficient), you can set a timeout for model training ([`max_model_t`](examples/config/halfcheetah/0.py#L30), in seconds) or train the model less frequently (every [`model_train_freq`](examples/config/halfcheetah/0.py#L22) steps).


### Acknowledge
The underlying soft actor-critic implementation in MBPO comes from Tuomas Haarnoja and Kristian Hartikainen's softlearning codebase. The modeling code is a slightly modified version of Janner's MBPO implementation.