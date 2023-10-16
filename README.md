# Unified Model Shift and Model Bias Policy Optimization

Code to reproduce the experiments in [How to Fine-tune the Model: Unified Model Shift and
Model Bias Policy Optimization] (NeuralIPS 2023).

If you choose to use this repo, please cite:

@article{zhang2023fine,
  title={How to Fine-tune the Model: Unified Model Shift and Model Bias Policy Optimization},
  author={Zhang, Hai and Yu, Hang and Zhao, Junqiao and Zhang, Di and Zhou, Hongtu and Zhang, Xiao and Ye, Chen and others},
  journal={arXiv preprint arXiv:2309.12671},
  year={2023}
}


## Installation
```
cd Unified-Model-Shift-and-Model-Bias-Policy-Optimization
conda env create -f environment/gpu-env.yml
conda activate usb
pip install -e .
```

## Usage
Configuration files can be found in [`examples/config/`](examples/config).

```
usb run_local examples.development --config=examples.config.halfcheetah.0 --gpus=1 --trial-gpus=1
```

Currently only running locally is supported.


#### Hyperparameters

The rollout length schedule is defined by a length-4 list in a [config file](examples/config/halfcheetah/0.py#L31). The format is `[start_epoch, end_epoch, start_length, end_length]`, so the following:
```
'rollout_schedule': [20, 100, 1, 5] 
```
corresponds to a model rollout length linearly increasing from 1 to 5 over epochs 20 to 100. 

If you want to speed up training in terms of wall clock time (but possibly make the runs less sample-efficient), you can set a timeout for model training ([`max_model_t`](examples/config/halfcheetah/0.py#L30), in seconds) or train the model less frequently (every [`model_train_freq`](examples/config/halfcheetah/0.py#L22) steps).


### Acknowledge
The underlying soft actor-critic implementation in MBPO comes from Tuomas Haarnoja and Kristian Hartikainen's softlearning codebase. The modeling code is a slightly modified version of Janner's MBPO implementation.