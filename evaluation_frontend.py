#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:04:19 2022

@author: brochetc


Metrics computation automation

"""


import os
import pickle
from collections import defaultdict
from functools import partial
from glob import glob
from multiprocessing import Pool

import base_config
import evaluation_backend as backend
import metrics4arome as metrics
import numpy as np
from configurate import Experiment

########### standard parameters #####

num_proc = base_config.num_proc
var_dict = base_config.var_dict 
real_data_dir = base_config.real_data_dir

#####################################


class EnsembleMetricsCalculator(Experiment):

    def __init__(self, expe_config, add_name):
        super().__init__(expe_config)

        self.add_name = add_name

    def __print__(self):
        super().__print__()
    ###########################################################################
    ######################### Main class method ###############################
    ###########################################################################

    def estimation(self, config, metrics_list, program, parallel=False, standalone=False, real=False):
        """

        estimate all metrics contained in metrics_list on training runs
        using specific strategies
                       -> parallel or sequential estimation
                       -> distance metrics or standalone metrics
                       -> on real samples only (if distance metrics)

        Inputs :

            metrics_list : list, the list of metrics to be computed

            program : dict of shape {int : (int, int)}
                      contains all the informations about sample numbers and number of repeats
                      #### WARNING : in this case, each sample is supposed to represent a given ensemble

                      keys index the repeats

                      values[0] index the type of dataset manipulation
                      (either dividing the same dataset into parts, or selecting only one portion)

                      values[1] indicate the number of samples to use in the computation

                      Note : -> for tests on training dynamics, only 1 repeat is assumed
                                  (at the moment)
                             -> for tests on self-distances on real datasets,
                                many repeats are possible (to account for test variance
                                or test different sampling sizes)

            parallel, standalone, real : bool, the flags defining the estimation
                                         strategy

        Returns :

            None

            dumps the results in a pickle file

        """
        # sanity checks
        if standalone and not parallel:
            raise ValueError("Estimation for standalone metric should be done in parallel")

        if standalone:
            assert set(metrics_list) <= metrics.standalone_metrics
        else:
            assert set(metrics_list) <= metrics.distance_metrics

        for metric in metrics_list:
            assert hasattr(metrics, metric)

        ########################
        self.program = program
        if parallel:
            if standalone:
                name = '_standalone_metrics_'
                if real:
                    def func(config, m_list): return self.parallelEstimation_standAlone(config, m_list, option='real')
                else:
                    func = self.parallelEstimation_standAlone
            else:
                name = '_distance_metrics_'
                if real:
                    func = self.parallelEstimation_realVSreal
                else:
                    func = self.parallelEstimation_realVSfake
                    
        else:
            name = '_distance_metrics_'
            if real:
                func = self.sequentialEstimation_realVSreal
            else:
                func = self.sequentialEstimation_realVSfake
        results = func(config, metrics_list)

        N_samples_set = sorted([n_samples for n_samples in self.program.values()])
        N_samples_name = '_'.join([str(n) for n in N_samples_set])

        if not real:
            N_samples_name = f"step_{'_'.join([str(s) for s in self.steps])}_{N_samples_name}"
        else:
            N_samples_name = f"{'_'.join([var for var in self.var_names])}_dom_{self.dom_size}_{N_samples_name}"
        dumpfile = f"{self.log_dir}{self.add_name}{name}{N_samples_name}.p"
        with open(dumpfile, 'wb') as pkfile:
            pickle.dump(results, pkfile)
        print(f"\n\n{'*' * 100}\nResults {'standalone' if standalone else 'distance'} {'real' if real else 'fake'} saved in {dumpfile}\n{'*' * 100}\n")

    ###########################################################################
    ############################   Estimation strategies ######################
    ###########################################################################
    def parallelEstimation_realVSfake(self, config, metrics_list):
        """

        makes a list of datasets with each item of self.steps
        and use multiple processes to evaluate the metric in parallel on each
        item.

        The metric must be a distance metric and the data should be real / fake

        Inputs : 

            metric : str, the metric to evaluate

        Returns :

            res : ndarray, the results array (precise shape defined by the metric)

        """
        print("PARALLEL ESTIMATION REALvsFAKE")
        RES = {}
        RES_inv = {}
        crop_size = crop_size = (self.CI[1] - self.CI[0], self.CI[3] - self.CI[2])
        Transformer = backend.Transform(config, crop_size)
        if len(self.steps) > len(self.program):
            for prog_idx in self.program.keys():
                data_list = [(config, Transformer, prog_idx, step, metrics_list) for step in self.steps]
                with Pool(num_proc) as p :
                    res = p.starmap(self.process_data, data_list)
                result_dict_list = {dict_step_tuple[1]: dict_step_tuple[0] for dict_step_tuple in res}
                RES[prog_idx] = result_dict_list
        else:
            for step in self.steps:
                data_list = [(config, Transformer, prog_idx, step, metrics_list, False) for prog_idx in self.program.keys()]
                with Pool(num_proc) as p :
                    res = p.starmap(self.process_data, data_list)
                result_dict_list = {dict_step_tuple[1]: dict_step_tuple[0] for dict_step_tuple in res}
                RES_inv[step] = result_dict_list
            RES = {}
            for step, dict_prog_idx_res in RES_inv.items():
                for prog_idx, result in dict_prog_idx_res.items():
                    if prog_idx in RES.keys():
                        RES[prog_idx][step] = result[0]
                    else:
                        RES[prog_idx] = {step: result[0]}
        print(f"RES.keys(): {RES.keys()}")
        print(f"RES[0].keys(): {RES[0].keys()}")
        print(f"RES[0][0].keys(): {RES[0][0].keys()}")
        return RES
    
    def process_data(self, config, Transformer, prog_idx, step, metrics_list, parallel_on_steps = True):
            dataset_r = backend.build_real_datasets(real_data_dir, self.program)[prog_idx]
            files = glob(os.path.join(self.data_dir_f,f"{self.fake_prefix}{step}_*.npy"))
            N_samples = self.program[prog_idx]
            N_samples = check_number_files(files, N_samples)
            result = backend.eval_distance_metrics(config, Transformer, metrics_list, {'real': dataset_r, 'fake': files}, N_samples, N_samples, self.VI, self.VI_f, self.CI, step)
            if parallel_on_steps:
                return result
            return result, prog_idx
    # def parallelEstimation_realVSfake(self, config, metrics_list):
    #     """

    #     makes a list of datasets with each item of self.steps
    #     and use multiple processes to evaluate the metric in parallel on each
    #     item.

    #     The metric must be a distance metric and the data should be real / fake

    #     Inputs : 

    #         metric : str, the metric to evaluate

    #     Returns :

    #         res : ndarray, the results array (precise shape defined by the metric)

    #     """
    #     mean_pert = self.mean_pert
    #     RES = {}
    #     RES_inv = {}
    #     Detransf = backend.Detransform(config)
    #     if len(self.steps) > len(self.program):
    #         for prog_idx in self.program.keys():
    #             data_list=[]
    #             for step in self.steps:
    #                 #getting first (and only) item of the random real dataset program
    #                 dataset_r = backend.build_datasets(real_data_dir, self.program, fake_prefix = self.fake_prefix)[prog_idx]
    #                 N_samples = self.program[prog_idx][1]
                    
    #                 #getting files to analyze from fake dataset
    #                 files = glob(f"{self.data_dir_f}{self.fake_prefix}{step}_*.npy")
    #                 N_samples = check_number_files(files, N_samples)
    #                 data_list.append((config, metrics_list, {'real': dataset_r, 'fake': files}, N_samples, N_samples, Detransf, self.VI, self.VI_f, self.CI, step))
    #             with Pool(num_proc) as p :
    #                 res = p.map(backend.eval_distance_metrics, data_list)
    #             result_dict_list = {dict_step_tuple[1]: dict_step_tuple[0] for dict_step_tuple in res}
    #             print(f"result_dict_list.keys(): {result_dict_list.keys()}")
    #             print(f"result_dict_list[{self.steps[0]}].keys(): {result_dict_list[self.steps[0]].keys()}")
    #             RES[prog_idx] = res
    #     else:
    #         for step in self.steps:
    #             data_list=[]
    #             for prog_idx in self.program.keys():
    #                 #getting first (and only) item of the random real dataset program
    #                 dataset_r = backend.build_datasets(real_data_dir, self.program, fake_prefix = self.fake_prefix)[prog_idx]
    #                 N_samples = self.program[prog_idx][1]
                    
    #                 #getting files to analyze from fake dataset
    #                 files = glob(f"{self.data_dir_f}{self.fake_prefix}{step}_*.npy")
    #                 N_samples = check_number_files(files, N_samples)
    #                 data_list.append((config, metrics_list, {'real': dataset_r, 'fake': files}, N_samples, N_samples, Detransf, self.VI, self.VI_f, self.CI, index))
    #             with Pool(num_proc) as p :
    #                 res = p.map(backend.eval_distance_metrics, data_list)
    #             result_dict_list = {dict_step_tuple[1]: dict_step_tuple[0] for dict_step_tuple in res}
    #             print(f"result_dict_list.keys(): {result_dict_list.keys()}")
    #             print(f"result_dict_list[{self.steps[0]}].keys(): {result_dict_list[self.steps[0]].keys()}")
    #             RES_inv[step] = res
    #         for key, value in RES_inv.items():
    #             for key_val, val in value.items():
    #                 d_interm = {}
    #                 d_interm[key] = value
    #                 RES[key_val] = d_interm
    #     return RES

    def sequentialEstimation_realVSfake(self, config, metrics_list, detransf=True):
        """

        Iterates the evaluation of the metric on each item of self.steps

        The metric must be a distance metric and the data should be real / fake

        Inputs : 

            metric : str, the metric to evaluate

        Returns :

            N_samples : int, the number of samples used in evaluation
            res : ndarray, the results array (precise shape defined by the metric)

        """
        mean_pert = self.mean_pert
        RES = {}
        if detransf:
            Detransf = backend.Detransform(config)
        else:
            Detransf = None
        for prog_idx in self.program.keys():
            res = []
            for step in self.steps:
                # getting first (and only) item of the random real dataset program
                dataset_r = backend.build_datasets(data_dir, self.program, fake_prefix=self.fake_prefix)[prog_idx]
                
                N_samples = self.program[prog_idx][1]
                # getting files to analyze from fake dataset
                files = glob(f"{self.data_dir_f}{self.fake_prefix}{step}_*.npy")

                data = (metrics_list, {'real': dataset_r, 'fake': files},
                        N_samples, N_samples,
                        self.VI, self.VI_f, self.CI, step)

                res.append(partial(backend.eval_distance_metrics(data), mean_pert=mean_pert, iter=self.iter))

            # some cuisine to produce a rightly formatted dictionary

            d_res = defaultdict(list)

            for res_index in res:
                res0 = res_index[0]
                for k, v in res0.items():
                    d_res[k].append(v)

            res = {k: np.concatenate([np.expand_dims(v[i], axis=0)
                                      for i in range(len(self.steps))], axis=0).squeeze()
                   for k, v in d_res.items()}

            RES[i0] = res

        if i0 == 0:

            return res
        else:
            return RES

    def parallelEstimation_realVSreal(self, config, metric):
        """

        makes a list of datasets with each pair of real datasets contained
        in self.program.

        Use multiple processes to evaluate the metric in parallel on each
        item.

        The metric must be a distance metric and the data should be real / real

        Inputs : 

            metric : str, the metric to evaluate

        Returns :

            N_samples : int, the number of samples used in evaluation
            res : ndarray, the results array (precise shape defined by the metric)

        """
        print("PARALLEL ESTIMATION REALvsREAL")
        crop_size = (self.CI[1] - self.CI[0], self.CI[3] - self.CI[2])
        Transformer = backend.Transform(config, crop_size)
        datasets_real = backend.build_real_datasets(real_data_dir, self.program, distance=True)
        data_list = [(config, Transformer, datasets_real, prog_idx, metric) for prog_idx in self.program.keys()]
        with Pool(num_proc) as p :
            res = p.starmap(self.process_data_real, data_list)
        # some cuisine to produce a rightly formatted dictionary
        print(res)
        result_dict_list = {dict_idxprog_tuple[1]: dict_idxprog_tuple[0] for dict_idxprog_tuple in res}
        return result_dict_list

    def process_data_real(self, config, Transformer, datasets, key_prog, metric):
        N_samples = self.program[key_prog]
        result = backend.eval_distance_metrics(config, Transformer, metric, {'real0': datasets[key_prog][0], 'real1': datasets[key_prog][1]}, N_samples, N_samples, self.VI, self.VI, self.CI, key_prog)
        return result

    def sequentialEstimation_realVSreal(self, metric, detransf=True):
        """

        Iterates the evaluation of the metric on each item of pair of real datasets 
        defined in self.program.

        The metric must be a distance metric and the data should be real / fake

        Inputs : 

            metric : str, the metric to evaluate

        Returns :

            N_samples : int, the number of samples used in evaluation
            res : ndarray, the results array (precise shape defined by the metric)

        """
            
        #getting first (and only) item of the random real dataset program
        
        datasets = backend.build_datasets(real_data_dir, self.program,
                                          self.real_dataset_labels,
                                          fake_prefix = self.fake_prefix)

        for i in range(len(datasets)):

            N_samples = self.program[i][1]

            data = (metric, {'real0': datasets[i][0], 'real1': datasets[i][1]},
                    N_samples, N_samples,
                    self.VI, self.VI, self.CI, i)

            if i == 0:
                res = [backend.eval_distance_metrics(data, mean_pert=self.mean_pert, iter=self.iter)]
            else:
                res.append(backend.eval_distance_metrics(data, mean_pert=self.mean_pert, iter=self.iter))

        # some cuisine to produce a rightly formatted dictionary

        d_res = defaultdict(list)

        for res_index in res:
            res0 = res_index[0]
            for k, v in res0.items():
                d_res[k].append(v)

        res = {k: np.concatenate([np.expand_dims(v[i], axis=0)
                                  for i in range(len(self.steps))], axis=0).squeeze()
               for k, v in d_res.items()}

        return res

    def parallelEstimation_standAlone(self, config, metrics_list, option='fake'):
        """

        makes a list of datasets with each dataset contained
        in self.program (case option =real) or directly from data files 
        (case option =fake)

        Use multiple processes to evaluate the metric in parallel on each
        item.

        The metric must be a standalone metric.

        Inputs : 

            metric : str, the metric to evaluate

        Returns :

            res : ndarray, the results array with shape [index of program]{"metric": [precise shape defined by the metric][variables][crop_size][crop_size]}

        """
        print(f"PARALLEL ESTIMATION STANDALONE {option}")
        RES = {}
        crop_size = (self.CI[1] - self.CI[0], self.CI[3] - self.CI[2])
        Transformer = backend.Transform(config, crop_size)
        if option == 'real':
            print("Multicomputing on programms")
            dataset_real_dict = backend.build_real_datasets(real_data_dir, self.program)
            data_list = [(config, Transformer, metrics_list, dataset, self.program[program_idx], self.VI, self.VI, self.CI, program_idx, option) for program_idx, dataset in dataset_real_dict.items()]
            
            with Pool(num_proc) as p :
                res = p.map(backend.global_dataset_eval, data_list)
            result_dict_list = {dict_idx_tuple[1]: dict_idx_tuple[0] for dict_idx_tuple in res}
            RES = {program_idx: {0: result} for program_idx, result in result_dict_list.items()}
            print(f"RES.keys(): {RES.keys()}")
            print(f"RES[0].keys(): {RES[0].keys()}")
            print(f"RES[0][0].keys(): {RES[0][0].keys()}")
            return RES

        elif option == 'fake':
            for prog_idx in self.program.keys():
                n_samples = self.program[prog_idx]
                data_list = []
                for step in self.steps:
                    # getting files to analyze from fake dataset
                    files = glob(f"{self.data_dir_f}{self.fake_prefix}{step}_*.npy")
                    n_samples = check_number_files(files, n_samples)

                    data_list.append((config, Transformer, metrics_list, files, n_samples, self.VI, self.VI_f, self.CI, step, option))
                with Pool(num_proc) as p:
                    res = p.map(backend.global_dataset_eval, data_list)
                result_dict_list = {dict_step_tuple[1]: dict_step_tuple[0] for dict_step_tuple in res}
                RES[prog_idx] = result_dict_list
            print(f"RES.keys(): {RES.keys()}")
            print(f"RES[0].keys(): {RES[0].keys()}")
            print(f"RES[0][0].keys(): {RES[0][0].keys()}")
            return RES

def check_number_files(files, n_samples):
    Shape = np.load(files[0], mmap_mode='c').shape
    if Shape[0] * len(files) < n_samples:
        raise ValueError(f"Not enough fakes to sample, you want {n_samples}, there are only {Shape[0] * len(files)} for files similar (same step) to {files[0]}")
    if n_samples % Shape[0] != 0:
        print(f"You provided a sample number {n_samples} that is not a multiple of batch size {Shape[0]}. Defaulting to the nearest smaller multiple : number = {(n_samples // Shape[0]) * Shape[0]}")
        n_samples = (n_samples // Shape[0]) * Shape[0]
    return n_samples