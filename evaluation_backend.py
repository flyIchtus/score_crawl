#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:54:05 2022

@author: brochetc

dataset metric tests
code snippets

"""
import sys
import os
if '/home/mrmn/brochetc/gan4arome' in sys.path :
    sys.path.remove('/home/mrmn/brochetc/gan4arome')
sys.path.append('/home/mrmn/poulainauzeaul/stylegan4arome/')


import numpy as np
import metrics4arome as metrics
from glob import glob
import random


########### standard parameters #####

num_proc = 8
var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' : 3 , 'orog' : 4, 'z500': 5, 't850': 6, 'tpw850': 7} # do not touch unless
                                                                                                # you know what u are doing
data_dir_0 = '/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
global_data_dir = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/"
print(os.path.exists(data_dir_0))
#####################################

def split_dataset(file_list,N_parts):
    """
    randomly separate a list of files in N_parts distinct parts
    
    Inputs :
        file_list : a list of filenames
        
        N_parts : int, the number of parts to split on
    
    Returns :
         list of N_parts lists of files
    
    """
    
    inds=[i*len(file_list)//N_parts for i in range(N_parts)]+[len(file_list)]

    to_split=file_list.copy()
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
    
    res= scale*(BigMat-Mean)/(Max)

    return  res


def build_datasets(data_dir, program, step=None, option='real'):
    """
    
    Build file lists to get samples, as specified in the program dictionary
    
    Inputs :
        
        data_dir : str, the directory to get the data from
        
        program : dict,the datasets to be constructed
                dict { dataset_id : (N_parts, n_samples)}
                
        step : None or int -> if None, normal search among generated samples
                              if int, search among generated samples at the given step
                              (used in learning dynamics mapping)
    
    Returns :
        
        res, dictionary of the shape {dataset_id : file_list}
        
        !!! WARNING !!! : the shape of file_list depends on the number of parts
        specified in the "program" items. Can be nested.
    
    """
    if step is not None:
        name='_Fsample_'+str(step)+'_'
    else:
        name='_Fsample'
        

    if option=='fake':
        globList=glob(data_dir+name+'*')
        
    else:
        globList=glob(data_dir+'_sample*')
        
    res={}
    
    
    for key, value in program.items():
        if value[0]==2:

            fileList=random.sample(globList,2*value[1])
            #print(len(fileList))
            
            res[key]=split_dataset(fileList,2)
                        
        if value[0]==1:
            
            fileList=random.sample(globList,value[1])
            
            res[key]=fileList

    return res


def load_batch(file_list,number,\
               var_indices_real = None, var_indices_fake = None,
               crop_indices=None,
               option='real', mean_pert = False, \
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
    
    if option=='fake':
        # in this case samples can either be in isolated files or grouped in batches
        
        assert var_indices_fake is not None # sanity check
        
        
        if len(var_indices_fake) ==1 :                    
            ind_var = var_indices_fake[0]
        
        Shape = np.load(file_list[0]).shape

        ## case : isolated files (no batching)
        
        if len(Shape)==3:
            
            Mat = np.zeros((number, len(var_indices_fake), Shape[1], Shape[2]), dtype=np.float32)
            
            list_inds = random.sample(file_list, number)
            
            for i in range(number) :
                
                if len(var_indices_fake)==1 :
                    if not mean_pert:
                        Mat[i] = np.load(list_inds[i])[ind_var:ind_var+1,:,:].astype(np.float32)
                    else:
                        # we need to denormalized in physical space then re-assemble into a single prev then renorm
                        var_idxs = [var_id for var_id in var_indices_real] +\
                                       [var_id+8 for var_id in var_indices_real]

                        Means_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                "mean_mean_pert.npy")[var_idxs].reshape(2,1,1)
                        Maxs_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                "max_mean_pert.npy")[var_idxs].reshape(2,1,1)
                        Stds_denorm = (1.0/0.95) * Maxs_denorm

                        Means_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "mean_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                        Maxs_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "max_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                        Stds_renorm = (1.0/0.95) * Maxs_renorm

                        ind_vars = [ind_var, ind_var+1]

                        tmp = np.load(list_inds[i])[ind_vars,:,:].astype(np.float32)*Stds_denorm+Means_denorm
                        # Then sum mean and perturbations
                        tmp2 = np.zeros((len(var_indices_fake), *tmp.shape[-2:]))
                        for j in range(tmp2.shape[0]):
                            tmp2[j,:,:] = tmp[j,:,:] + tmp[j+len(var_indices_fake),:,:]

                        # Renormalize 
                        Mat[i] = (tmp2-Means_renorm)/Stds_renorm
                else :
                    if not mean_pert:
                        Mat[i] = np.load(list_inds[i])[var_indices_fake,:,:].astype(np.float32)
                    else:
                        # we need to denormalized in physical space then re-assemble into a single prev then renorm

                        var_idxs = [var_id for var_id in var_indices_real] +\
                                       [var_id+8 for var_id in var_indices_real]

                        Means_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                "mean_mean_pert.npy")[var_idxs].reshape(len(var_idxs),1,1)
                        Maxs_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                "max_mean_pert.npy")[var_idxs].reshape(len(var_idxs),1,1)
                        Stds_denorm = (1.0/0.95) * Maxs_denorm

                        Means_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "mean_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                        Maxs_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "max_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                        Stds_renorm = (1.0/0.95) * Maxs_renorm

                        # First denormalize data
                        tmp = np.load(list_inds[i])[var_idxs,:,:].astype(np.float32)*Stds_denorm+Means_denorm
                        # Then sum mean and perturbations
                        tmp2 = np.zeros((len(var_indices_fake), *tmp.shape[-2:]))
                        for j in range(tmp2.shape[0]):
                            tmp2[j,:,:] = tmp[j,:,:] + tmp[j+len(var_indices_fake),:,:]

                        # Renormalize 
                        Mat[i] = (tmp2-Means_renorm)/Stds_renorm
        
        ## case : batching -> select the right number of files to get enough samples
        elif len(Shape)==4:
            print('loading batches')
            batch = Shape[0]
                        
            if batch > number : # one file is enough
                
                indices = random.sample(range(batch), number)
                k = random.randint(0,len(file_list)-1)
                
                if len(var_indices_fake)==1 :
                    if not mean_pert:
                        Mat = np.load(file_list[k])[indices, ind_var:ind_var+1,:,:]

                    else:
                        # we need to denormalized in physical space then re-assemble into a single prev then renorm
                        var_idxs = [var_id for var_id in var_indices_real] +\
                                       [var_id+8 for var_id in var_indices_real]

                        Means_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                "mean_mean_pert.npy")[var_idxs].reshape(2,1,1)
                        Maxs_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                "max_mean_pert.npy")[var_idxs].reshape(2,1,1)
                        Stds_denorm = (1.0/0.95) * Maxs_denorm

                        Means_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "mean_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                        Maxs_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "max_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                        Stds_renorm = (1.0/0.95) * Maxs_renorm

                        ind_vars = [ind_var, ind_var+1]

                        tmp = np.load(file_list[k])[indices, ind_vars,:,:].astype(np.float32)*Stds_denorm+Means_denorm
                        # Then sum mean and perturbations
                        tmp2 = np.zeros((tmp.shape[0], len(var_indices_fake), *tmp.shape[-2:]))
                        for j in range(tmp2.shape[1]):
                            tmp2[:,j,:,:] = tmp[:,j,:,:] + tmp[:,j+len(var_indices_fake),:,:]
                        
                        #print("a", tmp.min(axis=(0,2,3)), tmp.max(axis=(0,2,3)))
                        #print("1", tmp2.min(axis=(0,2,3)), tmp2.max(axis=(0,2,3)), tmp2.shape)
                        
                        # Renormalize 
                        Mat = (tmp2-Means_renorm)/Stds_renorm
                        #print("2", Mat.min(axis=(0,2,3)), Mat.max(axis=(0,2,3)), Mat.shape)
                else :
                    if not mean_pert:
                        Mat = np.load(file_list[k])[indices, var_indices_fake,:,:]

                    else:
                        # we need to denormalized in physical space then re-assemble into a single prev then renorm

                        var_idxs = [var_id for var_id in var_indices_real] +\
                                       [var_id+8 for var_id in var_indices_real]
                        ind_vars = var_indices_fake + [v_idx_f+len(var_indices_fake) for v_idx_f in var_indices_fake]

                        Means_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                "mean_mean_pert.npy")[var_idxs].reshape(len(var_idxs),1,1)
                        Maxs_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                "max_mean_pert.npy")[var_idxs].reshape(len(var_idxs),1,1)
                        Stds_denorm = (1.0/0.95) * Maxs_denorm

                        Means_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "mean_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                        Maxs_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "max_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                        Stds_renorm = (1.0/0.95) * Maxs_renorm

                        # First denormalize data
                        tmp = np.load(file_list[k])[indices, ind_vars,:,:].astype(np.float32)*Stds_denorm+Means_denorm
                        # Then sum mean and perturbations
                        tmp2 = np.zeros((tmp.shape[0], len(var_indices_fake), *tmp.shape[-2:]))
                        for j in range(tmp2.shape[1]):
                            tmp2[:,j,:,:] = tmp[:,j,:,:] + tmp[:,j+len(var_indices_fake),:,:]

                        
                        # Renormalize 
                        Mat = (tmp2-Means_renorm)/Stds_renorm
                
            
            else : #select multiple files and fill the number
            
                Mat=np.zeros((number, len(var_indices_fake), Shape[2], Shape[3]), \
                                                         dtype=np.float32)
                
                list_inds=random.sample(file_list, number//batch)
                
                for i in range(number//batch) :
                    
                    if len(var_indices_fake)==1 :
                        if not mean_pert:
                            Mat[i*batch:(i+1)*batch] = np.load(list_inds[i]).astype(np.float32)[:,ind_var:ind_var+1,:,:]
                        else:
                            # we need to denormalized in physical space then re-assemble into a single prev then renorm
                            var_idxs = [var_id for var_id in var_indices_real] +\
                                       [var_id+8 for var_id in var_indices_real]

                            Means_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                    "mean_mean_pert.npy")[var_idxs].reshape(2,1,1)
                            Maxs_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                    "max_mean_pert.npy")[var_idxs].reshape(2,1,1)
                            Stds_denorm = (1.0/0.95) * Maxs_denorm

                            Means_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "mean_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                            Maxs_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "max_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                            Stds_renorm = (1.0/0.95) * Maxs_renorm

                            ind_vars = [ind_var, ind_var+1]

                            tmp = np.load(list_inds[i])[:,ind_vars,:,:].astype(np.float32)*Stds_denorm+Means_denorm
                            # Then sum mean and perturbations
                            tmp2 = np.zeros((tmp.shape[0], len(var_indices_fake), *tmp.shape[-2:]))
                            for j in range(tmp2.shape[1]):
                                tmp2[:,j,:,:] = tmp[:,j,:,:] + tmp[:,j+len(var_indices_fake),:,:]
                            
                            #print("b", tmp.min(axis=(0,2,3)), tmp.max(axis=(0,2,3)))
                            #print("y", Maxs_renorm.item(), Means_renorm.item())
                            #print("3", tmp2.min(axis=(0,2,3)), tmp2.max(axis=(0,2,3)), tmp2.shape)
                            
                            # Renormalize 
                            Mat[i*batch :(i+1)*batch] = (tmp2-Means_renorm)/Stds_renorm
                            #print("4", Mat.min(axis=(0,2,3)), Mat.max(axis=(0,2,3)), Mat.shape)
                    else :
                        if not mean_pert:
                            Mat[i*batch: (i+1)*batch]=np.load(list_inds[i]).astype(np.float32)[:,var_indices_fake,:,:]
                        else:
                            var_idxs = [var_id for var_id in var_indices_real] +\
                                       [var_id+8 for var_id in var_indices_real]
                            ind_vars = var_indices_fake + [v_idx_f+len(var_indices_fake) for v_idx_f in var_indices_fake]

                            Means_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                    "mean_mean_pert.npy")[var_idxs].reshape(len(var_idxs),1,1)
                            Maxs_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                    "max_mean_pert.npy")[var_idxs].reshape(len(var_idxs),1,1)
                            Stds_denorm = (1.0/0.95) * Maxs_denorm

                            Means_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "mean_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                            Maxs_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "max_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                            Stds_renorm = (1.0/0.95) * Maxs_renorm

                            # First denormalize data
                            tmp = np.load(list_inds[i])[:, ind_vars,:,:].astype(np.float32)*Stds_denorm+Means_denorm
                            # Then sum mean and perturbations
                            tmp2 = np.zeros((tmp.shape[0], len(var_indices_fake), *tmp.shape[-2:]))
                            for j in range(tmp2.shape[1]):
                                tmp2[:,j,:,:] = tmp[:,j,:,:] + tmp[:,j+len(var_indices_fake),:,:]

                            # Renormalize 
                            Mat[i*batch : (i+1)*batch] = (tmp2-Means_renorm)/Stds_renorm
                    
                if number%batch !=0 :
                    
                    remain_inds = random.sample(range(batch),number%batch)
                    
                    if len(var_indices_fake)==1 :
                        if not mean_pert:
                            Mat[i*batch :] = np.load(list_inds[i+1])[remain_inds, ind_var:ind_var+1,:,:].astype(np.float32)
                        else:
                            # we need to denormalized in physical space then re-assemble into a single prev then renorm
                            var_idxs = [var_id for var_id in var_indices_real] +\
                                       [var_id+8 for var_id in var_indices_real]

                            Means_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                    "mean_mean_pert.npy")[var_idxs].reshape(2,1,1)
                            Maxs_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                    "max_mean_pert.npy")[var_idxs].reshape(2,1,1)
                            Stds_denorm = (1.0/0.95) * Maxs_denorm

                            Means_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "mean_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                            Maxs_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "max_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                            Stds_renorm = (1.0/0.95) * Maxs_renorm

                            ind_vars = [ind_var, ind_var+1]

                            tmp = np.load(list_inds[i+1])[remain_inds,ind_vars,:,:].astype(np.float32)*Stds_denorm+Means_denorm
                            # Then sum mean and perturbations
                            tmp2 = np.zeros((tmp.shape[0], len(var_indices_fake), *tmp.shape[-2:]))
                            for j in range(tmp2.shape[1]):
                                tmp2[:,j,:,:] = tmp[:,j,:,:] + tmp[:,j+len(var_indices_fake),:,:]

                            #print("c", tmp.min(axis=(0,2,3)), tmp.max(axis=(0,2,3)))
                            #print("5", tmp2.min(axis=(0,2,3)), tmp2.max(axis=(0,2,3)))
                            
                            # Renormalize 
                            Mat[i*batch :] = (tmp2-Means_renorm)/Stds_renorm
                            #print("6", Mat.min(axis=(0,2,3)), Mat.max(axis=(0,2,3)))
                    else :
                        if not mean_pert:
                            Mat[i*batch :] = np.load(list_inds[i+1])[remain_inds, var_indices_fake,:,:].astype(np.float32)
                        else:
                            var_idxs = [var_id for var_id in var_indices_real] +\
                                       [var_id+8 for var_id in var_indices_real]
                            ind_vars = var_indices_fake + [v_idx_f+len(var_indices_fake) for v_idx_f in var_indices_fake]

                            Means_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                    "mean_mean_pert.npy")[var_idxs].reshape(len(var_idxs),1,1)
                            Maxs_denorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"+\
                                    "max_mean_pert.npy")[var_idxs].reshape(len(var_idxs),1,1)
                            Stds_denorm = (1.0/0.95) * Maxs_denorm

                            Means_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "mean_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                            Maxs_renorm = np.load(global_data_dir + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"+\
                                "max_with_8_var.npy")[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
                            Stds_renorm = (1.0/0.95) * Maxs_renorm

                            # First denormalize data
                            tmp = np.load(list_inds[i+1])[remain_inds, ind_vars,:,:].astype(np.float32)*Stds_denorm+Means_denorm
                            # Then sum mean and perturbations
                            tmp2 = np.zeros((tmp.shape[0], len(var_indices_fake), *tmp.shape[-2:]))
                            for j in range(tmp2.shape[1]):
                                tmp2[:,j,:,:] = tmp[:,j,:,:] + tmp[:,j+len(var_indices_fake),:,:]

                            # Renormalize 
                            Mat[i*batch :] = (tmp2-Means_renorm)/Stds_renorm
                

    elif option=='real':
        
        # in this case samples are stored once per file
        Shape=(len(var_indices_real),
               crop_indices[1]-crop_indices[0],
               crop_indices[3]-crop_indices[2])
        
        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        list_inds=random.sample(file_list, number) # randomly drawing samples
        
        for i in range(number):
            if len(var_indices_real)==1 :
                
                ind_var = var_indices_real[0]
                
                Mat[i]=np.load(list_inds[i])[ind_var:ind_var+1,
                                           crop_indices[0]:crop_indices[1],
                                           crop_indices[2]:crop_indices[3]].astype(np.float32)
            else :
                
                Mat[i]=np.load(list_inds[i])[var_indices_real,
                                           crop_indices[0]:crop_indices[1],
                                           crop_indices[2]:crop_indices[3]].astype(np.float32)
                

    return Mat



def eval_distance_metrics(data, option = 'from_names', mean_pert=False,
                    mean_file='mean_with_8_var.npy',
                    max_file='max_with_8_var.npy'):
    
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
    metrics_list, dataset, n_samples_0, n_samples_1, VI, VI_f, CI, index, real_dir = data
    ## loading and normalizing data
    
    Means=np.load(data_dir_0+mean_file)[VI].reshape(1,len(VI),1,1)
    Maxs=np.load(data_dir_0+max_file)[VI].reshape(1,len(VI),1,1)
    
    print('Loading data: ', list(dataset.keys()))
    if list(dataset.keys())==['real','fake']:
    
        print('index',index)
    
        if option=='from_names':
            
            assert(type(dataset['fake'])==list)
            
            fake_data = load_batch(dataset['fake'], n_samples_1, var_indices_real = VI, var_indices_fake = VI_f, 
                                   option='fake', mean_pert=mean_pert)
            #fake_data = load_batch(dataset['real'], n_samples_0, var_indices_real = VI, var_indices_fake = VI_f, crop_indices = CI)
            #fake_data = normalize(fake_data, 0.95, Means, Maxs)
            print('fake data loaded')
        if option=='from_matrix':
           
            assert(type(dataset['fake']==str))
            
            fake_data = np.load(dataset['fake'], dtype=np.float32)
        
        print('loading real data')
        
        real_data = load_batch(dataset['real'], n_samples_0, var_indices_real = VI, var_indices_fake = VI_f, crop_indices = CI)
        
        print('normalizing')
        real_data = normalize(real_data, 0.95, Means, Maxs)
        #print(real_data.max(axis=(0,2,3)), real_data.min(axis=(0,2,3)))
    
    elif list(dataset.keys())==['real0', 'real1']:
    
        print(index)
    
        real_data0 = load_batch(dataset['real0'], n_samples_0, var_indices_real = VI, crop_indices = CI)
        real_data1 = load_batch(dataset['real1'], n_samples_1, var_indices_real= VI, crop_indices = CI)
        
        real_data = normalize(real_data0, 0.95, Means, Maxs)
        fake_data = normalize(real_data1, 0.95, Means, Maxs)  # not stricly "fake" but same
        
    else :
        raise ValueError("Dataset keys must be either 'real'/'fake' or 'real0'/'real1', not {}"
                         .format(list(dataset.keys())))
        
    ## the interesting part : computing each metric of metrics_list
    
    results = {}
    
    for metric in metrics_list :
    
        print(metric)
        
        Metric = getattr(metrics, metric)
    
        results[metric] = Metric(real_data, fake_data, select = False) 
    
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
    
    metrics_list, dataset, n_samples, VI, VI_f, CI, index, data_option, real_dir = data
    print('evaluating backend',index)
    #print(real_dir)
    if option=="from_names":
        
        if data_option=='fake' :
            
            assert(type(dataset)==list)
            print('loading fake data')
            rdata = load_batch(dataset, n_samples, var_indices_real=VI, var_indices_fake=VI_f, crop_indices = CI, 
                               option=data_option, mean_pert=mean_pert)

        elif data_option=='real' :
           
           assert(type(dataset[0])==list)
           print('loading real data') 
           rdata = load_batch(dataset[0], n_samples,var_indices_real = VI,crop_indices = CI,option=data_option)

    if option=='from_matrix':
       
        assert(type(dataset)==str)
        
        rdata = np.load(dataset, dtype=np.float32)
    
    if data_option=='real':
        print('normalizing')    
        Means=np.load(real_dir+mean_file)[VI].reshape(1,len(VI),1,1)
        Maxs=np.load(real_dir+max_file)[VI].reshape(1,len(VI),1,1)
    
        rdata=normalize(rdata,0.95,Means, Maxs)
    
   
    
    
    results = {}
    
    for metric in metrics_list :
    
        print(metric)
        
        Metric = getattr(metrics, metric)
        
        results[metric] = Metric(rdata, select = False)
    return results, index

