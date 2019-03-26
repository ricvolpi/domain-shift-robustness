#!/bin/bash

#python main.py --mode=test_all --gpu=$1 --exp_dir=./pretrained-models/ERM
#python main.py --mode=test_all --gpu=$1 --exp_dir=./pretrained-models/RDA
#python main.py --mode=test_all --gpu=$1 --exp_dir=./pretrained-models/RSDA
python main.py --mode=test_all --gpu=$1 --exp_dir=./pretrained-models/ESDA
