#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:11:52 2022

@author: brochetc

Metrics Executable

"""
import evaluation_frontend as frontend
from configurate import getAndNameDirs, select_Config


if __name__ == "__main__":

    configuration_set, N_samples = getAndNameDirs(option='rigid')

    program = {i: (1, N_samples) for i in range(1)}

    distance_metrics_list = ["pw_W1", "multivar", "W1_random_NUMPY",
                             "W1_Center_NUMPY", "SWD_metric_torch", "quant_metric"]
    # standalone_metrics_list=["spectral_compute", "struct_metric","ls_metric"]
    standalone_metrics_list = ["spectral_compute", "ls_metric"]

    for ind in range(configuration_set.length):

        expe_config = select_Config(configuration_set, ind, option='rigid')

        mC = frontend.EnsembleMetricsCalculator(
            expe_config, add_name='test_standalone')

        mC.estimation(standalone_metrics_list, program,
                      standalone=True, parallel=True)

        mC = frontend.EnsembleMetricsCalculator(
            expe_config, add_name='test_distance', )

        mC.estimation(distance_metrics_list, program,
                      standalone=False, parallel=True)