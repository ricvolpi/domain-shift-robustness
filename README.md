# Code for the paper [Model Vulnerability over Image Transformation Sets](https://github.com/ricvolpi/domain-shift-robustness)

## Overview

The code in this repo allows to test the vulnerability of (black-box) models via random search and evolution search over arbitrary transformation sets, and train more robust models via the RDA/RSDA/ESDA algorithms presented in the paper. The current version only allows to train/test MNIST models, the rest of the code will be released ASAP.

### Files

``model.py``: to build tf's graph

``train_ops.py``: train/test functions

``search_ops.py``: search algos (RS/ES from the paper)

``transformations_ops.py``: modules to build image transformation set and apply transformations

``exp_config``: config file with the hyperparameters

Some pretrained models are included in the ''pretrained-models'' folder, with the associated ``exp_config`` files.

### Prerequisites

Python 2.7, Tensorflow 1.12.0

## How it works

To obtain MNIST and SVHN dataset, run

```
mkdir data
python download_and_process_mnist.py
sh download_svhn.sh
```
##
To train the model, run

```
python main.py --mode=train_MODE --gpu=GPU_IDX --exp_dir=EXP_DIR
```
where MODE can be one of {ERM, RDA, RSDA, ESDA}, GPU_IDX is the index of the GPU to be used, and EXP_DIR is the folder containing the exp_config file.

##
To run evolution search (ES) or random search (RS) on a trained model, run

```
python main.py --mode=test_MODE --gpu=GPU_IDX --exp_dir=EXP_DIR
```
where MODE can be one of {RS, ES}. For ES, population size POP_SIZE and mutation rate ETA can be set as
 
```
python main.py --mode=test_ES --gpu=GPU_IDX --exp_dir=EXP_DIR --pop_size=POP_SIZE --mutation_rate=ETA
```

##
To test performance on all digit datasets (MNIST, SVHN, MNIST-M, SYN, USPS), run

```
python main.py --mode=test_all --gpu=GPU_IDX --exp_dir=EXP_DIR
```
MNIST-M, SYN and USPS testing are currently commented out, uncomment them when you have downloaded the datasets. Loading code is included in train_ops.py.

##

If one desires to include more transformations, or explore different magnitude ranges of the provided one, modifications to transformations_ops.py should be straightforward. Please, let me know if anything doesn't work, or if you have any useful feedback! 




 
