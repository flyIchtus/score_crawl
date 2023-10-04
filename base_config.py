#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:23:17 2023

@author: brochetc
"""

########### standard parameters #####

num_proc = 8
var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4} # do not touch unless
                                                          # you know what u are doing
real_data_dir = '/scratch/mrmn/gandonb/data/cropped_120_376_540_796/'
mean_pert_data_dir = '/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/IS_1_1.0_0_0_0_0_0_256_mean_pert/'
fake_prefix = '_Fsample_'

real_dataset_labels = 'labels.csv' #'IS_train_dataset.csv'

if real_dataset_labels == 'labels.csv':
    num = 1
else:
    num = 0
mean_pert_data_file = "mean_mean_pert.npy"
max_pert_data_file = "max_mean_pert.npy"

mean_data_file = 'stats_file/mean_rr_log.npy'
max_data_file = 'stats_file/std_rr_log.npy'
#####################################
