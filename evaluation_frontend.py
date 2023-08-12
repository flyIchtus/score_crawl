#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:04:19 2022

@author: brochetc


Metrics computation automation

"""


import pickle
from glob import glob
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
from functools import partial


from configurate import Experiment
import evaluation_backend as backend
import metrics4arome as metrics



########### standard parameters #####

num_proc = backend.num_proc
var_dict = backend.var_dict
data_dir = backend.data_dir_0

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

    def estimation(self, metrics_list, program, parallel=False, standalone=False, real=False):
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
            raise (
                ValueError, 'Estimation for standalone metric should be done in parallel')

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

                    def func(m_list): return self.parallelEstimation_standAlone(
                        m_list, option='real')

                else:
                    func = self.parallelEstimation_standAlone

            else:

                name = '_distance_metrics_'

                if real:
                    func = self.parallelEstimation_realVreal
                else:
                    func = self.parallelEstimation_realVSfake
        else:

            name = '_distance_metrics_'

            if real:
                func = self.sequentialEstimation_realVSreal

            else:
                func = self.sequentialEstimation_realVSfake

        results = func(metrics_list)

        N_samples_set = [self.program[i][1] for i in range(len(program))]

        N_samples_name = '_'.join([str(n) for n in N_samples_set])

        if not real:
            N_samples_name = 'step_' + '_'.join([str(s) for s in self.steps]) + '_' + N_samples_name
        else:
            N_samples_name = '_'.join([var for var in self.var_names]) + f'_dom_{self.dom_size}_' + N_samples_name

        if real:

            temp_log_dir = self.log_dir

            self.log_dir = backend.data_dir_0

        dumpfile = self.log_dir + self.add_name+name + str(N_samples_name)+'.p'

        if real:

            self.log_dir = temp_log_dir

        pickle.dump(results, open(dumpfile, 'wb'))

    ###########################################################################
    ############################   Estimation strategies ######################
    ###########################################################################

    def parallelEstimation_realVSfake(self, metrics_list):
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
        mean_pert = self.mean_pert
        print("mean/pert", mean_pert)
        RES = {}
        for step in self.steps:
            print('Step', step)
            data_list = []
            for i0 in self.program.keys():

                # getting first (and only) item of the random real dataset program
                dataset_r = backend.build_datasets(data_dir, self.program,
                                                   real_prefix=self.real_prefix,
                                                   fake_prefix=self.fake_prefix)[i0]

                N_samples = self.program[i0][1]

                # getting files to analyze from fake dataset
                files = glob(self.data_dir_f +
                             self.fake_prefix + str(step)+'_*.npy')

                data_list.append((metrics_list, {'real': dataset_r, 'fake': files},
                                  N_samples, N_samples,
                                  self.VI, self.VI_f, self.CI, step, data_dir))

            with Pool(num_proc) as p:
                res = p.map(partial(backend.eval_distance_metrics, mean_pert=mean_pert), data_list)

            # some cuisine to produce a rightly formatted dictionary

            ind_list=[]
            d_res = defaultdict(list)
            
            for res_index in res :
                
                index = res_index[1]
                res0 = res_index[0]
                for k, v in res0.items():
                    
                    d_res[k].append(v)
                ind_list.append(index)
            
            #for k in d_res.keys():
            #    print((ind_list, d_res[k]))
            #    d_res[k]= [x for _,x in sorted(zip(ind_list, d_res[k]))]
            
            res = { k : v  for k,v in d_res.items()}
            RES[step] = res
        if len(self.steps)==1:
            RES2 = {}
            for i0 in self.program.keys():
                RES2[i0] = {}
                for key in RES[step].keys():
                    RES2[i0][key] = RES[step][key][i0]
        if step==self.steps[0] and i0==0:
            return res
        else :
            return RES if len(self.steps)!=1 else RES2

    def sequentialEstimation_realVSfake(self, metrics_list):
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

        for i0 in self.program.keys():

            res = []

            for step in self.steps:

                # getting first (and only) item of the random real dataset program
                dataset_r = backend.build_datasets(data_dir, self.program,
                                                   real_prefix=self.real_prefix,
                                                   fake_prefix=self.fake_prefix)[i0]

                N_samples = self.program[i0][1]

                # getting files to analyze from fake dataset
                files = glob(self.data_dir_f +
                             self.fake_prefix + str(step)+'_*.npy')

                data = (metrics_list, {'real': dataset_r, 'fake': files},
                        N_samples, N_samples,
                        self.VI, self.VI_f, self.CI, step, data_dir)

                res.append(partial(backend.eval_distance_metrics(data), mean_pert=mean_pert))

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

    def parallelEstimation_realVSreal(self, metric):
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

        datasets = backend.build_datasets(data_dir, self.program,
                                          real_prefix=self.real_prefix,
                                          fake_prefix=self.fake_prefix)
        data_list = []

        # getting the two random datasets programs

        for i in range(len(datasets)):

            N_samples = self.program[i][1]

            data_list.append((metric,
                              {'real0': datasets[i][0],
                                  'real1': datasets[i][1]},
                              N_samples, N_samples,
                              self.VI, self.VI, self.CI, i, data_dir))

        with Pool(num_proc) as p:
            res = p.map(backend.eval_distance_metrics, data_list)

        # some cuisine to produce a rightly formatted dictionary

        ind_list = []
        d_res = defaultdict(list)

        for res_index in res:
            index = res_index[1]
            res0 = res_index[0]
            for k, v in res0.items():
                d_res[k].append(v)
            ind_list.append(index)

        for k in d_res.keys():
            d_res[k] = [x for _, x in sorted(zip(ind_list, d_res[k]))]

        res = {k: np.concatenate([np.expand_dims(v[i], axis=0)
                                  for i in range(len(self.steps))], axis=0).squeeze()
               for k, v in d_res.items()}

        return res

    def sequentialEstimation_realVSreal(self, metric):
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

        # getting first (and only) item of the random real dataset program

        datasets = backend.build_datasets(data_dir, self.program,
                                          real_prefix=self.real_prefix,
                                          fake_prefix=self.fake_prefix)

        for i in range(len(datasets)):

            N_samples = self.program[i][1]

            data = (metric, {'real0': datasets[i][0], 'real1': datasets[i][1]},
                    N_samples, N_samples,
                    self.VI, self.VI, self.CI, i, data_dir)

            if i == 0:
                res = [backend.eval_distance_metrics(data)]
            else:
                res.append(backend.eval_distance_metrics(data))

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

    def parallelEstimation_standAlone(self, metrics_list, option='fake'):
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

            res : ndarray, the results array (precise shape defined by the metric)

        """

        mean_pert = self.mean_pert
        if option == 'real':

            self.steps = [0]
            dataset_r = backend.build_datasets(data_dir, self.program,
                                               real_prefix=self.real_prefix,
                                               fake_prefix=self.fake_prefix)

            data_list = [(metrics_list, dataset_r, self.program[i][1],
                          self.VI, self.VI, self.CI, i, option, data_dir)
                         for i, dataset in enumerate(dataset_r)]

            with Pool(num_proc) as p:

                res = p.map(backend.global_dataset_eval, data_list)

            ind_list = []
            d_res = defaultdict(list)

            for res_index in res:
                index = res_index[1]
                res0 = res_index[0]
                for k, v in res0.items():
                    d_res[k].append(v)
                ind_list.append(index)

            for k in d_res.keys():
                d_res[k] = [x for _, x in sorted(zip(ind_list, d_res[k]))]

            res = {k: np.concatenate([np.expand_dims(v[i], axis=0)
                                      for i in range(len(self.steps))], axis=0).squeeze()
                   for k, v in d_res.items()}

            return res

        elif option == 'fake':

            RES = {}
            for i0 in self.program.keys():

                N_samples = self.program[i0][1]
                data_list = []

                for j, step in enumerate(self.steps):

                    # getting files to analyze from fake dataset
                    if option == 'fake':
                        files = glob(self.data_dir_f +
                                     self.fake_prefix + str(step)+'_*.npy')

                        data_list.append((metrics_list, files, N_samples,
                                          self.VI, self.VI_f, self.CI, step, option, data_dir))

                with Pool(num_proc) as p:
                    res = p.map(partial(backend.global_dataset_eval, mean_pert=mean_pert), data_list)

                ind_list = []
                d_res = defaultdict(list)

                for res_index in res:
                    index = res_index[1]
                    res0 = res_index[0]
                    for k, v in res0.items():
                        d_res[k].append(v)
                    ind_list.append(index)

                for k in d_res.keys():
                    d_res[k] = [x for _, x in sorted(zip(ind_list, d_res[k]))]

                res = {k: np.concatenate([np.expand_dims(v[i], axis=0)
                                          for i in range(len(self.steps))], axis=0).squeeze()
                       for k, v in d_res.items()}

                RES[i0] = res

            if i0 == 0:
                return res
            return RES
