# Code for the paper [Model Vulnerability over Image Transformation Sets](https://github.com/ricvolpi/domain-shift-robustness)

## Overview

### Files

``model.py``: to build tf's graph

``train_ops.py``: train/test functions

``search_ops.py``: search algos (RS/ES from the paper)

``transformations_ops.py``: modules to build image transformation set and apply transformations

``exp_configuration``: config file with the hyperparameters

### Prerequisites

Python 2.7, Tensorflow 1.6.0

## How it works

To obtain MNIST and SVHN dataset, run

```
mkdir data
python download_and_process_mnist.py
sh download_svhn.sh
```

To train the model, run

```
sh run_exp.sh GPU_IDX
```

where GPU_IDX is the index of the GPU to be used.

 
