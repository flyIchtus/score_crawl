#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:11:52 2022

@author: brochetc

Metrics Executable

"""
import base_config
import evaluation_frontend as frontend
from configurate import getAndNameDirs, select_Config

if __name__ == "__main__":
    print(f"CPU used by this programm: {base_config.num_proc}")
    configuration_set, N_samples = getAndNameDirs(option='flex')

    program = {i: N_samples for i in range(base_config.repeat)}

    num = base_config.num
    for idx in range(configuration_set.length):
        config = select_Config(configuration_set, idx, option='flex')
        # ## STANDALONE
        metrics_str = '_'.join(base_config.standalone_metrics_list)
        mC = frontend.EnsembleMetricsCalculator(config, add_name=f"{base_config.prefix}_{metrics_str}_{num}")
        mC.estimation(config, base_config.standalone_metrics_list, program, standalone=True, parallel=True, real=base_config.real)

        ## DISTANCE
        metrics_str = '_'.join(base_config.distance_metrics_list)
        mC = frontend.EnsembleMetricsCalculator(config, add_name=f"{base_config.prefix}_{metrics_str}_{num}")
        mC.estimation(config, base_config.distance_metrics_list, program, standalone=False, parallel=True, real=base_config.real)
