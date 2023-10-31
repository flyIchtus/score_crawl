#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:23:17 2023

@author: brochetc
"""

########### standard parameters #####
import os

num_proc = os.cpu_count()
num_proc = 4

prefix = "METR_clip"

real = False
repeat = 1


var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4} # do not touch unless
                                                          # you know what u are doing
root_expe_path = "/scratch/work/gandonb/Exp_StyleGAN/"
exp_config_dir = f"/home/gmap/mrmn/gandonb/SAVE/styleganPNRIA/gan/configs/"
real_data_dir = '/scratch/work/gandonb/data/cropped_120_376_540_796/'
mean_pert_data_dir = '/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/IS_1_1.0_0_0_0_0_0_256_mean_pert/'
fake_prefix = '_Fsample_'

data_transform_config_filename = f"dataset_handler_config.yaml"
real_dataset_labels = 'labels.csv' #'IS_train_dataset.csv'

if real_dataset_labels == 'labels.csv':
    num = 1
else:
    num = 0

mean_pert_data_file = "mean_mean_pert.npy"
max_pert_data_file = "max_mean_pert.npy"
########## METRICS ##########
standalone_metrics_list = ["spectral_compute", "ls_metric", "quant_map"]
# standalone_metrics_list = ["spectral_compute", "ls_metric"] # short
# standalone_metrics_list = ["quant_map"]

distance_metrics_list = ["W1_random_NUMPY", "W1_Center_NUMPY", "SWD_metric_torch"]
# distance_metrics_list = ["W1_random_NUMPY", "W1_Center_NUMPY"] # short
# distance_metrics_list = ["SWD_metric_torch"]


#####################################
