#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:54:05 2022

@author: brochetc

dataset metric tests
code snippets

"""
import numpy as np
import metrics4arome as metrics
from glob import glob
import random
import base_config
import pandas as pd

real_data_dir = base_config.real_data_dir
mean_pert_data_dir = base_config.mean_pert_data_dir


def split_dataset(file_list, N_parts):
    """
    randomly separate a list of files in N_parts distinct parts

    Inputs :
        file_list : a list of filenames

        N_parts : int, the number of parts to split on

    Returns :
         list of N_parts lists of files

    """
    
    inds = [i*len(file_list)//N_parts for i in range(N_parts)]+[len(file_list)]

    to_split = file_list.copy()
    random.shuffle(to_split)

    return [to_split[inds[i]:inds[i+1]] for i in range(N_parts)]


def normalize(BigMat, scale, Mean, Max):
    """

    Normalize samples with specific Mean and max + rescaling

    Inputs :

        BigMat : ndarray, samples to rescale

        scale : float, scale to set maximum amplitude of samples

        Mean, Max : ndarrays, must be broadcastable to BigMat

    Returns :

        res : ndarray, same dimensions as BigMat

    """

    res = scale*(BigMat-Mean)/(Max)

    return res

##################################

def mean_pert_rescale(filename, 
                        data_dir_mean_pert, data_dir_physical, 
                        var_indices_real, var_indices_fake, 
                        indices = None, case = 'isolated_single'):

    """ intricate rescaling to deal with perturbations and mean splitted generation
    Set loaded data in a normalized [-1,1] interval

    list_inds :  filename list, used to load splitted data files
    Mat : receiving np.ndarray, to store final results

    data_dir_mean_pert : str, where to fetch splitted normalization constants
    data_dir_physical :  str, where to fetch physical constants
    var_indices_real : indices of the real variables to be fetched in real norm constants
    var_indices_fake : indices of the fake variables to be fetched in fake data samples
    case : str, discusses the formatting options to handle indices correctly

    """
	var_idxs = [var_id for var_id in var_indices_real] +\
		                               [var_id+8 for var_id in var_indices_real]

	Means_denorm = np.load(data_dir_mean_pert + "mean_mean_pert.npy")[var_idxs].reshape(2,1,1)
	Maxs_denorm = np.load(data_dir_mean_pert + "max_mean_pert.npy")[var_idxs].reshape(2,1,1)
	Stds_denorm = (1.0/0.95) * Maxs_denorm

	Means_renorm = np.load(data_dir_physical + \
		"mean_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
	Maxs_renorm = np.load(data_dir_physical + \
		"max_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
		
	Stds_renorm = (1.0/0.95) * Maxs_renorm

    if case=='isolated_single': # files are not batched, single variable

        ind_vars = [ind_var, ind_var+1]

        tmp = np.load(list_inds[i])[ind_vars,:,:].astype(np.float32) * Stds_denorm + Means_denorm
        # Then sum mean and perturbations
        tmp2 = np.zeros((len(var_indices_fake), *tmp.shape[-2:]))
        for j in range(tmp2.shape[0]):
            tmp2[j,:,:] = tmp[j,:,:] + tmp[j+len(var_indices_fake),:,:]

    elif case=='isolated_multiple':  # files are not batched, multiple variables

        # First denormalize data
        tmp = np.load(list_inds[i])[var_idxs,:,:].astype(np.float32) * Stds_denorm + Means_denorm
        # Then sum mean and perturbations
        tmp2 = np.zeros((len(var_indices_fake), *tmp.shape[-2:]))
        for j in range(tmp2.shape[0]):
            tmp2[j,:,:] = tmp[j,:,:] + tmp[j+len(var_indices_fake),:,:]

    elif case=='batched_single': # here indices is needed

        ind_vars = [ind_var, ind_var+1]
        tmp0 =  np.load(file_list[k])[indices]
        tmp = tmp0[:, ind_vars,:,:].astype(np.float32) * Stds_denorm + Means_denorm
        # Then sum mean and perturbations
        tmp2 = np.zeros((tmp.shape[0], len(var_indices_fake), *tmp.shape[-2:]))
        for j in range(tmp2.shape[1]):
            tmp2[:,j,:,:] = tmp[:,j,:,:] + tmp[:,j+len(var_indices_fake),:,:]

    elif case=='batched_multiple': # here indices is needed

        ind_vars = var_indices_fake + [v_idx_f+len(var_indices_fake) for v_idx_f in var_indices_fake]
        # First denormalize data
        tmp0 =  np.load(file_list[k])[indices]
        tmp = tmp0[:,ind_vars,:,:].astype(np.float32) * Stds_denorm + Means_denorm
        # Then sum mean and perturbations
        tmp2 = np.zeros((tmp.shape[0], len(var_indices_fake), *tmp.shape[-2:]))
        for j in range(tmp2.shape[1]):
            tmp2[:,j,:,:] = tmp[:,j,:,:] + tmp[:,j+len(var_indices_fake),:,:]


    else:
        raise ValueError('Unknown case {}'.format(case))

	# Renormalize 
	mat = (tmp2-Means_renorm)/Stds_renorm
	
	return mat

##################################

def build_datasets(data_dir, program, step=None, option='real',
                   fake_prefix='_Fsample_', real_prefix='_sample'):
    """

    Build file lists to get samples, as specified in the program dictionary

    Inputs :

        data_dir : str, the directory to get the data from

        program : dict,the datasets to be constructed
                dict { dataset_id : (N_parts, n_samples)}

        step : None or int -> if None, normal search among generated samples
                              if int, search among generated samples at the given step
                              (used in training steps mapping)
    
    Returns :

        res, dictionary of the shape {dataset_id : file_list}

        !!! WARNING !!! : the shape of file_list depends on the number of parts
        specified in the "program" items. Can be nested.

    """
    if step is not None:
        name = fake_prefix + str(step)+'_'
    else:
        name = fake_prefix

    if option=='fake':
        globList = glob(data_dir + name + '*')
        
    else:
        
        print('reading dataframe', dataframe)
        
        df = pd.read_csv(data_dir + dataframe)
        
        globList = [data_dir + filename + '.npy' for filename in df['Name']]
  
    res = {}

    for key, value in program.items():
        if value[0] == 2:

            fileList = random.sample(globList,2*value[1])
            
            res[key] = split_dataset(fileList,2)
                        
        if value[0] == 1:
            
            fileList = random.sample(globList,value[1])
            
            res[key] = fileList

    return res


def load_batch(file_list, number,
               var_indices_real=None, var_indices_fake=None,
               crop_indices=None,
               option='real', mean_pert=False,
               output_dir=None, output_id=None, save=False):
    """
    gather a fixed number of random samples present in file_list into a single big matrix

    Inputs :

        file_list : list of files to be sampled from

        number : int, the number of samples to draw

        var_indices(_real/_fake) : iterable of ints, coordinates of variables in a given sample to select

        crop_indices : iterable of ints, coordinates of data to be taken (only in 'real' mode)

        Shape : tuple, the target shape of every sample

        option : str, different treatment if the data is GAN generated or PEARO
        
        mean_pert : if True, we will add the mean and pert that were recorded separetely before computing any score
        
        output_dir : str of the directory to save datasets ---> NotImplemented

        output_id : name of the dataset to be saved ---> NotImplemented

        save : bool, whether or not to save the loaded dataset ---> NotImplemented

    Returns :

        Mat : numpy array, shape  number x C x Shape[1] x Shape[2] matrix

    """

    print('length of file list', len(file_list))

    if option == 'fake':
        # in this case samples can either be in isolated files or grouped in batches

        assert var_indices_fake is not None  # sanity check

        if len(var_indices_fake) == 1:
            ind_var = var_indices_fake[0]

        Shape = np.load(file_list[0]).shape

        # case : isolated files (no batching)

        if len(Shape) == 3:

            Mat = np.zeros((number, len(var_indices_fake),
                           Shape[1], Shape[2]), dtype=np.float32)

            list_inds = random.sample(file_list, number)

            for i in range(number):

                if len(var_indices_fake) == 1:
                    if not mean_pert:
                        Mat[i] = np.load(list_inds[i])[
                            ind_var:ind_var+1, :, :].astype(np.float32)
                    else:
                        Mat[i] = mean_pert_rescale(list_inds[i], 
                        mean_pert_data_dir,
                        real_data_dir,
                        var_indices_real, var_indices_fake, case='isolated_single') 
                 
                else:
                    if not mean_pert:
                        Mat[i] = np.load(list_inds[i])[
                        var_indices_fake, :, :].astype(np.float32)
                    else:
                        Mat[i] = mean_pert_rescale(list_inds[i],
                        mean_pert_data_dir,
                        real_data_dir,
                        var_indices_real, var_indices_fake, case='isolated_mutiple') 

        # case : batching -> select the right number of files to get enough samples
        elif len(Shape) == 4:
            print('loading batches')
            batch = Shape[0]
            
            if batch > number:  # one file is enough

                indices = random.sample(range(batch), number)
                k = random.randint(0, len(file_list)-1)

                if len(var_indices_fake) == 1:
                    if not mean_pert:
                        Mat = np.load(file_list[k])[
                            indices, ind_var:ind_var+1, :, :]
                    else:
                        Mat = mean_pert_rescale(file_list[k],
                        mean_pert_data_dir,
                        real_data_dir,
                        var_indices_real,var_indices_fake, indices=indices, case='batched_single')
                else:
                    if not mean_pert :
                        Mat = np.load(file_list[k])[
                            indices, var_indices_fake, :, :]
                    else:
                        Mat = mean_pert_rescale(file_list[k],
                            mean_pert_data_dir,
                            real_data_dir,
                            var_indices_real, var_indices_fake, indices=indices, case='batched_multiple')
                
            else:  # select multiple files and fill the number of samples using these files

                Mat = np.zeros((number, len(var_indices_fake), Shape[2], Shape[3]),
                               dtype=np.float32)

                list_inds = random.sample(file_list, number//batch)

                for i in range(number//batch):

                    if len(var_indices_fake) == 1:
                        if not mean_pert:
                            Mat[i*batch: (i+1)*batch] =\
                                np.load(list_inds[i]).astype(np.float32)[
                                :, ind_var:ind_var+1, :, :]
                        else:
                            Mat[i*batch: (i+1)*batch] = mean_pert_rescale(list_inds[i], 
                                mean_pert_data_dir,
                                real_data_dir,
                                var_indices_real, var_indices_fake, case='batched_single')

                    else:
                        if not mean_pert:
                            Mat[i*batch: (i+1)*batch] =\
                                np.load(list_inds[i]).astype(np.float32)[
                                :, var_indices_fake, :, :]
                        else:
                            Mat[i*batch: (i+1)*batch] = mean_pert_rescale(list_inds[i], 
                                mean_pert_data_dir,
                                real_data_dir,
                                var_indices_real, var_indices_fake, case='batched_multiple')


                if number % batch != 0:

                    remain_inds = random.sample(range(batch), number % batch)

                    if len(var_indices_fake) == 1:
                        if not mean_pert:
                            tmp = np.load(
                                    list_inds[i+1])[remain_inds]
                            Mat[i*batch:] =\
                                tmp[:, ind_var:ind_var+1, :, :].astype(np.float32)
                        else:
                            Mat[i*batch:] = mean_pert_rescale(list_inds[i+1], 
                            mean_pert_data_dir,
                            real_data_dir,
                            var_indices_real, var_indices_fake, indices=remain_inds, case='batched_single')

                    else:
                        if not mean_pert:
                            tmp = np.load(
                                    list_inds[i+1])[remain_inds]
                            Mat[i*batch:] =\
                                tmp[:, var_indices_fake, :, :].astype(np.float32)
                        else:
                            Mat[i*batch:] = mean_pert_rescale(list_inds[i+1], 
                            mean_pert_data_dir,
                            real_data_dir,
                            var_indices_real, var_indices_fake, indices=remain_inds, case='batched_multiple')

    elif option == 'real':

        # in this case samples are stored once per file

        Shape = (len(var_indices_real),
                 crop_indices[1]-crop_indices[0],
                 crop_indices[3]-crop_indices[2])

        print("Shape", Shape)
        print("var_indices_real", var_indices_real)
        print("crop_indices", crop_indices)

        Mat = np.zeros(
            (number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)

        # randomly drawing samples
        list_inds = random.sample(file_list, number)

        for i in range(number):
            if len(var_indices_real) == 1:

                ind_var = var_indices_real[0]

                Mat[i] = np.load(list_inds[i])[ind_var:ind_var+1,
                                               crop_indices[0]:crop_indices[1],
                                               crop_indices[2]:crop_indices[3]].astype(np.float32)
            else:

                Mat[i] = np.load(list_inds[i])[var_indices_real,
                                               crop_indices[0]:crop_indices[1],
                                               crop_indices[2]:crop_indices[3]].astype(np.float32)

    return Mat

def eval_distance_metrics(data, option='from_names', mean_pert=False):
    """

    this function should test distance metrics for datasets=[(filelist1, filelist2), ...]
    in order to avoid memory overhead, datasets are created and destroyed dynamically

    Inputs :

       data : tuple of 

           metric : str, the name of a metric in the metrics4arome namespace

           dataset : 
               dict of file lists (option='from_names') /str (option 'from_matrix')

               Identifies the file names to extract data from

               Keys of the dataset are either : 'real', 'fake' when comparing 
                               sets of real and generated data

                                                'real0', 'real1' when comparing
                               different sets of real data

           n_samples_0, n_samples_1 : int,int , the number of samples to draw
                                      to compute the metric.

                                      Note : most metrics require equal numbers

                                      by default, 0 : real, 1 : fake

           VI, CI : iterables of indices to select (VI) variables/channels in samples
                                                   (CI) crop indices in maps

           index : int, identifier to the data passed 
                   (useful only if used in multiprocessing)


       option : str, to choose if generated data is loaded from several files (from_names)
               or from one big Matrix (from_matrix)

    Returns :

        results : np.array containing the calculation of the metric

    """

    metrics_list, dataset, n_samples_0, n_samples_1, VI, VI_f, CI, index = data
    
    ## loading and normalizing data
    
    Means = np.load(
        real_data_dir + 'mean_with_orog.npy')[VI].reshape(1,len(VI),1,1)
    Maxs = np.load(
        real_data_dir + 'max_with_orog.npy')[VI].reshape(1,len(VI),1,1)
    
    print('Loading data') 
    if list(dataset.keys()) == ['real','fake']:
    
        print('index',index)
    
        if option=='from_names':
            assert(type(dataset['fake']) == list)
            
            fake_data = load_batch(
                dataset['fake'], n_samples_1, var_indices_fake = VI_f, option='fake', mean_pert=mean_pert)
            
            print('fake data loaded')
        if option == 'from_matrix':

            assert (type(dataset['fake'] == str))

            fake_data = np.load(dataset['fake'], dtype=np.float32)

        print('loading real data')
        print('loading real n_samples_0 ', n_samples_0)
        print('loading real VI ', VI)
        print('loading real VI_f ', VI_f)
        print('loading real CI ', CI)

        real_data = load_batch(
            dataset['real'], n_samples_0, var_indices_real=VI, var_indices_fake=VI_f, crop_indices=CI)

        print('normalizing')
        real_data = normalize(real_data, 0.95, Means, Maxs)

    elif list(dataset.keys()) == ['real0', 'real1']:

        print(index)

        real_data0 = load_batch(
            dataset['real0'], n_samples_0, var_indices_real=VI, crop_indices=CI)
        real_data1 = load_batch(
            dataset['real1'], n_samples_1, var_indices_real=VI, crop_indices=CI)

        real_data = normalize(real_data0, 0.95, Means, Maxs)
        # not stricly "fake" but same
        fake_data = normalize(real_data1, 0.95, Means, Maxs)

    else:
        raise ValueError("Dataset keys must be either 'real'/'fake' or 'real0'/'real1', not {}"
                         .format(list(dataset.keys())))

    # the interesting part : computing each metric of metrics_list

    results = {}

    for metric in metrics_list:

        print(metric)

        Metric = getattr(metrics, metric)

        results[metric] = Metric(real_data, fake_data, select=False)

    return results, index

def global_dataset_eval(data, option='from_names',
                    mean_file='mean_with_8_var.npy',
                    max_file='max_with_8_var.npy', mean_pert=False):
    """

    evaluation of metric on the DataSet (treated as a single numpy matrix)

    Inputs :

        data : iterable (tuple)of str, dict, int

            metric : str, the name of a metric in the metrics4arome namespace

            dataset :
                file list /str containing the ids of the files to get samples


            n_samples_0, n_samples_1 : int,int , the number of samples to draw
                                          to compute the metric.

                                          Note : most metrics require equal numbers

                                          by default, 0 : real, 1 : fake

               VI, CI : iterables of indices to select (VI) variables/channels in samples
                                                       (CI) crop indices in maps

               index : int, identifier to the data passed 
                       (useful only if used in multiprocessing)


         option : str, to choose if generated data is loaded from several files (from_names)
                   or from one big Matrix (from_matrix)

    Returns :

        results : dictionary contining the metrics list evaluation

        index : the index input (to keep track on parallel execution)

    """

    metrics_list, dataset, n_samples, VI, VI_f, CI, index, data_option = data
    
    print('evaluating backend',index)
    #print(real_dir)
    if option=="from_names":
        
        if data_option=='fake' :
            
            assert(type(dataset)==list)
            
            print('loading fake data')
            rdata = load_batch(
                dataset, n_samples, var_indices_fake=VI_f, crop_indices=CI,
                option=data_option, mean_pert=mean_pert)

        elif data_option == 'real':

            assert (type(dataset[0]) == list)
            print('loading real data')
            rdata = load_batch(
                dataset[0], n_samples, var_indices_real=VI, crop_indices=CI, option=data_option)

    if option == 'from_matrix':

        assert (type(dataset) == str)

        rdata = np.load(dataset, dtype=np.float32)

    if data_option == 'real':
        print('normalizing')
        Means = np.load(
            real_dir+'mean_with_orog.npy')[VI].reshape(1, len(VI), 1, 1)
        Maxs = np.load(
            real_dir+'max_with_orog.npy')[VI].reshape(1, len(VI), 1, 1)
        rdata = normalize(rdata, 0.95, Means, Maxs)
        
    results = {}

    for metric in metrics_list:

        print(metric)

        Metric = getattr(metrics, metric)

        results[metric] = Metric(rdata, select=False)
    return results, index
