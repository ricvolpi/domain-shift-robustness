# Code for the paper [Addressing Model Vulnerability to Distributional Shifts over Image Transformation Sets](http://openaccess.thecvf.com/content_ICCV_2019/html/Volpi_Addressing_Model_Vulnerability_to_Distributional_Shifts_Over_Image_Transformation_Sets_ICCV_2019_paper.html)

## Overview

The code in this repo allows 

1. Testing the vulnerability of (black-box) models via random search and evolution search over arbitrary transformation sets. 

2. Training more robust models via the RDA/RSDA/ESDA algorithms presented in the paper. 

Here a small ConvNet and the MNIST dataset are used, but applying these tools to arbitrary tasks/models is straightforward. Feel free to drop me a message if any feedback can be helpful.

### Files

``model.py``: to build tf's graph

``train_ops.py``: train/test functions

``search_ops.py``: search algos (RS/ES from the paper)

``transformations_ops.py``: modules to build image transformation set and apply transformations

``exp_config``: config file with the hyperparameters

Some pretrained models are available in a [heavier version](https://github.com/ricvolpi/domain-shift-robustness-models) of this repo.

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
Testing MNIST-M, SYN and USPS is commented out.

##
If one desires to include more transformations, or explore different intensity intervals, modifications to transformations_ops.py should be straightforward. 

## Reference

**Addressing Model Vulnerability to Distributional Shifts over Image Transformation Sets**  
Riccardo Volpi and Vittorio Murino, ICCV 2019
```
    @InProceedings{Volpi_2019_ICCV,
    author = {Volpi, Riccardo and Murino, Vittorio},
    title = {Addressing Model Vulnerability to Distributional Shifts Over Image Transformation Sets},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
    }
```


 
