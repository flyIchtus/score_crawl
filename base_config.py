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
real_data_dir = '/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/'
mean_pert_dir = '/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/IS_1_1.0_0_0_0_0_0_256_mean_pert/'
fake_prefix = '_Fsample_'

real_dataset_labels = 'IS_train_dataset.csv'

#####################################