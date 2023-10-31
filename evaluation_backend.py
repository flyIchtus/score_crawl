#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:54:05 2022

@author: brochetc

dataset metric tests
code snippets

"""
import warnings

warnings.filterwarnings("error")

import asyncio
import concurrent.futures
import os
import random
import threading
import copy
from glob import glob

import base_config
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import metrics4arome as metrics
import numpy as np
import pandas as pd
import scipy

real_data_dir = base_config.real_data_dir
dataframe = base_config.real_dataset_labels
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
    inds = [i * len(file_list) // N_parts for i in range(N_parts)] + [len(file_list)]

    to_split = file_list.copy()
    random.shuffle(to_split)

    return [to_split[inds[i]:inds[i + 1]] for i in range(N_parts)]


class Transform:
    def __init__(self, config, crop_size):
        self.config = config
        self.crop_size = crop_size
        self.dataset_handler_yaml = config.data_transform_config
        self.maxs, self.mins, self.means, self.stds = self.init_normalization()
        if self.stds is not None:
            self.stds *= 1.0 / 0.95

    def init_normalization(self):
        normalization_type = self.dataset_handler_yaml["normalization"]["type"]
        if normalization_type == "mean":
            means, stds = self.load_stat_files(normalization_type, "mean", "std")
            return None, None, means, stds
        elif normalization_type == "minmax":
            maxs, mins = self.load_stat_files(normalization_type, "max", "min")
            return maxs, mins, None, None
        else:
            print("No normalization set")
            return None, None, None, None

    def load_stat_files(self, normalization_type, str1, str2):
        stat_version = self.dataset_handler_yaml["stat_version"]
        log_iterations = self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]
        per_pixel = self.dataset_handler_yaml["normalization"]["per_pixel"]

        mean_or_max_filename = f"{str1}_{stat_version}"
        mean_or_max_filename += "_log" * log_iterations
        std_or_min_filename = f"{str2}_{stat_version}"
        std_or_min_filename += "_log" * log_iterations

        if per_pixel:
            mean_or_max_filename += "_ppx"
            std_or_min_filename += "_ppx"
        mean_or_max_filename += ".npy"
        std_or_min_filename += ".npy"
        print(f"Normalization set to {normalization_type}")
        stat_folder = self.dataset_handler_yaml["stat_folder"]
        file_path = os.path.join(real_data_dir, stat_folder, mean_or_max_filename)
        means_or_maxs = np.load(file_path).astype('float32')
        print(f"{str1} file found")

        file_path = os.path.join(real_data_dir, stat_folder, std_or_min_filename)
        stds_or_mins = np.load(file_path).astype('float32')
        print(f"{str2} file found")
        return means_or_maxs, stds_or_mins

    def detransform(self, data, step):
        norm_type = self.dataset_handler_yaml["normalization"]["type"]
        per_pixel = self.dataset_handler_yaml["normalization"]["per_pixel"]
        rr_transform = self.dataset_handler_yaml["rr_transform"]
        if rr_transform["symetrization"]:
            self.mins = -self.maxs
            self.means = np.zeros_like(self.means)
        if norm_type == "mean":
            if not per_pixel:
                data = data * self.stds[np.newaxis, :, np.newaxis, np.newaxis] + self.means[np.newaxis, :, np.newaxis, np.newaxis]
            else:
                data = data * self.stds + self.means
        elif norm_type == "minmax":
            if not per_pixel:
                data = ((data + 1) / 2) * (self.maxs[np.newaxis, :, np.newaxis, np.newaxis] - self.mins[np.newaxis, :, np.newaxis, np.newaxis]) + self.mins[np.newaxis, :, np.newaxis, np.newaxis]
            else:
                data = ((data + 1) / 2) * (self.maxs - self.mins) + self.mins
        if rr_transform["symetrization"]:
            data[:, 0] = np.abs(data[:, 0])
        for _ in range(rr_transform["log_transform_iteration"]):
            try:
                data[:, 0] = np.exp(data[:, 0]) - 1
            except RuntimeWarning as error:
                print(f"RuntimeWarning for step {step}, in np.exp(data[:, 0]) - 1.")
        if rr_transform["gaussian_std"] > 0:
            mask_no_rr = data[:, 0] > rr_transform["gaussian_std"] * (1 + 0.25)
            data[:, 0] *= mask_no_rr
        print("Detransform OK.")
        return data

    def print_data_detransf(self, data, step):
        numpy_files = [os.path.join(real_data_dir, x) for x in os.listdir(real_data_dir) if x.endswith(".npy")]
        random_files = np.random.choice(numpy_files, size=4, replace=False)
        reals = np.array([np.load(file) for file in random_files])
        data_to_print = np.concatenate((data[:12], reals), axis=0)
        save_dir = f"{self.config.data_dir_f[:-1]}_detranformed/"
        os.makedirs(save_dir, exist_ok=True)

        print("Saving fake samples...")
        np.save(f"{save_dir}Samples_at_Step_{step}.npy", data[:12])

        print("Printing data...")
        self.online_sample_plot(data_to_print, step)

        print("Data printed.")
    
    def transform(self, data):
        means, stds, maxs, mins = copy.deepcopy(self.means), copy.deepcopy(self.stds), copy.deepcopy(self.maxs), copy.deepcopy(self.mins)
        norm_type = self.dataset_handler_yaml["normalization"]["type"]
        rr_transform = self.dataset_handler_yaml["rr_transform"]
        for _ in range(rr_transform["log_transform_iteration"]):
            data[:, 0] = np.log1p(data[:, 0])
        if rr_transform["symetrization"] and np.random.random() <= 0.5:
            data[:, 0] = -data[:, 0]
        if norm_type != "None":
            if rr_transform["symetrization"]: #applying transformations on rr only if selected
                if norm_type == "means":
                    means[0] = np.zeros_like(means[0])
                elif norm_type == "minmax":
                    mins[0] = -maxs[0]
        # gaussian_std = rr_transform["gaussian_std"]
        # if gaussian_std:
        #     for _ in range(rr_transform["log_transform_iteration"]):
        #         gaussian_std = np.log(1 + gaussian_std)
        #     gaussian_std_map = np.random.choice([-1, 1], size=self.crop_size) * gaussian_std
        #     gaussian_noise = np.mod(np.random.normal(0, gaussian_std, size=self.crop_size), gaussian_std_map)
        if norm_type == "mean":
            if np.ndim(stds) > 1:
                if self.dataset_handler_yaml["normalization"]["for_rr"]["blur_iteration"] > 0:
                    gaussian_filter = np.float32([[1, 4,  6,  4,  1],
                                                [4, 16, 24, 16, 4],
                                                [6, 24, 36, 24, 6],
                                                [4, 16, 24, 16, 4],
                                                [1, 4,  6,  4,  1]]) / 256.0
                    for _ in range(self.dataset_handler_yaml["normalization"]["for_rr"]["blur_iteration"]):
                        stds[0] = scipy.ndimage.convolve(stds[0], gaussian_filter, mode='mirror')
            else:
                means = means[np.newaxis, :, np.newaxis, np.newaxis]
                stds = stds[np.newaxis, :, np.newaxis, np.newaxis]
        elif norm_type == "minmax":
            print(maxs.shape)
            if np.ndim(maxs) > 1:
                if self.dataset_handler_yaml["normalization"]["for_rr"]["blur_iteration"] > 0:
                    gaussian_filter = np.float32([[1, 4,  6,  4,  1],
                                                [4, 16, 24, 16, 4],
                                                [6, 24, 36, 24, 6],
                                                [4, 16, 24, 16, 4],
                                                [1, 4,  6,  4,  1]]) / 256.0
                    for _ in range(self.dataset_handler_yaml["normalization"]["for_rr"]["blur_iteration"]):
                        maxs[0] = scipy.ndimage.convolve(maxs[0], gaussian_filter, mode='mirror')
            else:
                maxs = maxs[np.newaxis, :, np.newaxis, np.newaxis]
                mins = mins[np.newaxis, :, np.newaxis, np.newaxis]
        # if gaussian_std != 0:
        #     mask_no_rr = (data[:, 0] <= gaussian_std)
        #     data[:, 0] = data[:, 0] + gaussian_noise * mask_no_rr
        if norm_type == "means":
            data = (data - means) / stds
        elif norm_type == "minmax":
            data = -1 + 2 * ((data - mins) / (maxs - mins))
        print("Real samples transformed...")
        return data
        

    def online_sample_plot(self, batch, Step, mean_pert=False):
        bounds = np.array([0, 0.5, 1, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 1000])
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=18)
        cmapRR = colors.ListedColormap(["white", "#63006e", "#0000ff", "#00b2ff", "#00ffff", "#08dfd6", "#1cb8a5", "#6ba530", "#ffff00", "#ffd800", "#ffa500", "#ff0000", "#991407", "#ff00ff", "#a4ff00", "#00fa00", "#31be00", "#31858b"], name="from_list", N=None)
        batch_to_print = batch[:16]
        IMG_SIZE = batch.shape[2]
        
        variable_mapping = {
            "rr": ("Rain rate", cmapRR, (0, 1000)),
            "u": ("Wind u", "viridis", (-20, 20)),
            "v": ("Wind v", "viridis", (-20, 20)),
            "t2m": ("2m temperature", "coolwarm", (240, 316)),
            "orog": ("Orography", "terrain", (-0.95, 0.95)),
            "z500": ("500 hPa geopotential", "Blues", (0, 100)),
            "t850": ("850 hPa temperature", "coolwarm", (-0.5, 0.5)),
            "tpw850": ("tpw850", "plasma", (-0.5, 0.5)),
        }

        for i, var in enumerate(self.config.variables):
            varname, cmap, limits = variable_mapping.get(var)

            fig, axs = plt.subplots(4, 4, figsize=(20, 20))
            st = fig.suptitle(f"{varname}{' pert' if mean_pert else ''}", fontsize='30')
            st.set_y(0.96)

            for j, ax in enumerate(axs.ravel()):
                b = batch_to_print[j][i]
                if var == "rr":
                    im = ax.imshow(b[::-1, :], cmap=cmap, norm=norm)
                else:
                    im = ax.imshow(b[::-1, :], cmap=cmap, vmin=limits[0], vmax=limits[1])

            fig.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.9)
            cbax = fig.add_axes([0.92, 0.05, 0.02, 0.85])
            cb = fig.colorbar(im, cax=cbax)
            cb.ax.tick_params(labelsize=20)

            save_dir = f"{self.config.data_dir_f[:-1]}_detranformed/"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}Samples_at_Step_{Step}_{var}{'_pert' if mean_pert else ''}.png")
            plt.close()

            if mean_pert:
                fig, axs = plt.subplots(4, 4, figsize=(20, 20))
                st = fig.suptitle(f"{varname} mean", fontsize='30')
                st.set_y(0.96)

                for j, ax in enumerate(axs.ravel()):
                    b = batch_to_print[j][i + len(self.config.variables)].view(IMG_SIZE, IMG_SIZE)
                    if var == "rr":
                        im = ax.imshow(b[::-1, :], cmap=cmap, norm=norm)
                    else:
                        im = ax.imshow(b[::-1, :], cmap=cmap, vmin=limits[0], vmax=limits[1])

                fig.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.9)
                cbax = fig.add_axes([0.92, 0.05, 0.02, 0.85])
                cb = fig.colorbar(im, cax=cbax)
                cb.ax.tick_params(labelsize=20)

                plt.savefig(f"{save_dir}Samples_at_Step_{Step}_{var}{'_mean' if mean_pert else ''}.png")
                plt.close()

##################################

def mean_pert_rescale(filename, data_dir_mean_pert, data_dir_physical, var_indices_real, var_indices_fake, indices = None, case = 'isolated_single'):

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
    var_idxs = [var_id for var_id in var_indices_real] + [var_id+8 for var_id in var_indices_real]
    Means_denorm = np.load(data_dir_mean_pert + mean_pert_data_file)[var_idxs].reshape(2,1,1)
    Maxs_denorm = np.load(data_dir_mean_pert + max_pert_data_file)[var_idxs].reshape(2,1,1)
    Stds_denorm = (1.0/0.95) * Maxs_denorm

    Means_renorm = np.load(data_dir_physical + \
                  mean_data_file)[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
    Maxs_renorm = np.load(data_dir_physical + \
                  mean_data_file)[[var_id for var_id in var_indices_real]].reshape(len(var_indices_real),1,1)
		
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
def build_real_datasets(data_dir, program, distance=False):
    """
    Build file lists to get samples, as specified in the program dictionary

    Inputs :
        data_dir : str, the directory to get the data from
        program : dict, the datasets to be constructed
                  dict { dataset_id : (N_parts, n_samples)}
        step : None or int -> if None, normal search among generated samples
                              if int, search among generated samples at the given step
                              (used in training steps mapping)

    Returns :
        res, dictionary of the shape {dataset_id : file_list}
        !!! WARNING !!! : the shape of file_list depends on the number of parts
        specified in the "program" items. Can be nested.
    """
    print(f"reading real sample dataframe: {dataframe}")
    df = pd.read_csv(os.path.join(data_dir, dataframe))
    glob_list = [os.path.join(data_dir, f"{filename}.npy") for filename in df["Name"]]

    res = {}
    for program_idx, n_samples in program.items():
        if distance:
            file_list = random.sample(glob_list, 2 * n_samples)
            file_list = [file_list[:len(file_list) // 2], file_list[len(file_list) // 2:]]
        else:
            file_list = random.sample(glob_list, n_samples)
        res[program_idx] = file_list
    return res


def load_batch(file_list, number, var_indices_real=None, var_indices_fake=None, crop_indices=None, option='real', mean_pert=False, output_dir=None, output_id=None, save=False):
    """
    gather a fixed number of random samples present in file_list into a single big matrix

    Inputs :

        file_list : list of files to be sampled from

        number : int, the number of samples to draw

        var_indices(_real/_fake) : iterable of ints, coordinates of variables in a given sample to select

        crop_indices : iterable of ints, coordinates of data to be taken (only in 'real' mode)

        option : str, different treatment if the data is GAN generated or PEARO
        
        mean_pert : if True, we will add the mean and pert that were recorded separetely before computing any score
        
        output_dir : str of the directory to save datasets ---> NotImplemented

        output_id : name of the dataset to be saved ---> NotImplemented

        save : bool, whether or not to save the loaded dataset ---> NotImplemented

    Returns :

        Mat : numpy array, shape  number x C x Shape[1] x Shape[2] matrix

    """
    print(f"length of file list: {len(file_list)}")
    if option == 'fake':
        # in this case samples can either be in isolated files or grouped in batches
        assert var_indices_fake is not None  # sanity check
        Shape = np.load(file_list[0]).shape
        print(f"Shape of file: {Shape}")
        print(f"var_indices_fake: {var_indices_fake}")
        print(f"crop_indices: {crop_indices}")
        
        if len(Shape) == 3:  # Isolated files (no batching)
            Mat = np.zeros((number, len(var_indices_fake), Shape[1], Shape[2]), dtype=np.float32)
            list_inds = random.sample(file_list, number)
            
            for i in range(number):
                if not mean_pert:
                    Mat[i] = np.load(list_inds[i])[list(var_indices_fake), :, :].astype(np.float32)
                else:
                    Mat[i] = mean_pert_rescale(list_inds[i], mean_pert_data_dir, real_data_dir, var_indices_real, var_indices_fake, case='isolated_single')
         
        elif len(Shape) == 4:  # Batching -> select the right number of files to get enough samples
            print('Loading batches')
            batch = Shape[0]
            
            if batch >= number:  # One file is enough
                indices = random.sample(range(batch), number)
                k = random.randint(0, len(file_list) - 1)
                
                if not mean_pert:
                    Mat = (np.load(file_list[k]).astype(np.float32)[indices])[:, list(var_indices_fake), crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]
                else:
                    Mat = mean_pert_rescale(file_list[k], mean_pert_data_dir, real_data_dir, var_indices_real, var_indices_fake, indices=indices, case='batched_multiple')
            else:  # Select multiple files and fill the number of samples using these files
                Mat = np.empty((number, Shape[1], Shape[2], Shape[3]), dtype=np.float32)
                list_inds = random.sample(file_list, number // batch)
                
                for idx_inds, inds in enumerate(list_inds):
                    if not mean_pert:
                        Mat[idx_inds * batch:(idx_inds + 1) * batch] = np.load(inds).astype(np.float32)[:, var_indices_fake, crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]
                    else:
                        Mat[idx_inds * batch:(idx_inds + 1) * batch] = mean_pert_rescale(inds, mean_pert_data_dir, real_data_dir, var_indices_real, var_indices_fake, case='batched_multiple')
    elif option == 'real':
        Shape = (len(var_indices_real), crop_indices[1] - crop_indices[0], crop_indices[3] - crop_indices[2])
        print(f"Shape of files: {Shape}")
        print(f"var_indices_real: {var_indices_real}")
        print(f"crop_indices: {crop_indices}")

        # Initialize Mat as an empty array
        Mat = np.empty((number, *Shape), dtype=np.float32)

        try:
            if number > len(file_list) or number < 0:
                raise ValueError(f"Issue with the population. number ({number}) larger than population ({len(file_list)}) or negative.")
            print(f"Loading batch of {number} files")
            threads = []
            for file_idx, file_npy in enumerate(file_list):
                thread = threading.Thread(target=load_npy, args=(file_npy, file_idx, var_indices_real, crop_indices, Mat))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
        except ValueError as e:
            print(e)

    print(f"Mat shape: {Mat.shape}")
    return Mat

def load_npy(sample_name, sample_num, var_indices_real, crop_indices, Mat):
    Mat[sample_num] = np.load(sample_name)[var_indices_real, crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]].astype(np.float32)
"""                 
    elif option == 'real':
        Shape = (len(var_indices_real), crop_indices[1] - crop_indices[0], crop_indices[3] - crop_indices[2])
        print(f"Shape: {Shape}")
        print(f"var_indices_real: {var_indices_real}")
        print(f"crop_indices: {crop_indices}")

        # Initialize Mat as an empty array
        Mat = np.empty((number, *Shape), dtype=np.float32)

        try:
            if number > len(file_list) or number < 0:
                raise ValueError(f"Issue with the population. number ({number}) larger than population ({len(file_list)}) or negative.")
            
            list_inds = random.sample(file_list, number)
            print(f"Loading batch of {number} files")
            for sample_num, sample_name in enumerate(list_inds):
                if sample_num % 1000 == 0:
                    print(f"{sample_num} files loaded over {number} files...")
                
                Mat[sample_num] = np.load(sample_name).astype(np.float32)[var_indices_fake, crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3]]

        except ValueError as e:
            print(e)
    print(f"Mat shape: {Mat.shape}")
    return Mat"""

def eval_distance_metrics(config, Transformer, metrics_list, dataset, n_samples_0, n_samples_1, VI, VI_f, CI, step, option='from_names', mean_pert=False):
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
       
       mean_pert :  wether mean/perturbation splitting is used in fake data

       iter :  whether to perform exponential tf

    Returns :

        results : np.array containing the calculation of the metric

    """
    print(f"evaluating backend distances for step: {step}")
    ## loading and normalizing data
    print('Loading data')
    if list(dataset.keys()) == ['real', 'fake']:
        if option == 'from_names':
            assert(type(dataset['fake']) == list)
            fake_data = load_batch(dataset['fake'], n_samples_1, var_indices_fake=VI_f, crop_indices=CI, option='fake', mean_pert=mean_pert)
            print('fake data loaded')
            fake_data = Transformer.detransform(fake_data, step)
            # fake_data[:, 0] = np.clip(fake_data[:, 0], a_min=None, a_max=350)
            Transformer.print_data_detransf(fake_data, step)
            
        if option == 'from_matrix':
            assert (type(dataset['fake'] == str))
            fake_data = np.load(dataset['fake'], dtype=np.float32)
            

        print(f"n_samples real: {n_samples_0}")
        print(f"VI real: {VI}")
        print(f"VI fake: {VI_f}")
        print(f"CI real: {CI}")

        real_data = load_batch(dataset['real'], n_samples_0, var_indices_real=VI, var_indices_fake=VI_f, crop_indices=CI)
        # real_data = Transformer.transform(real_data)

    elif list(dataset.keys()) == ['real0', 'real1']:
        real_data0 = load_batch(dataset['real0'], n_samples_0, var_indices_real=VI, crop_indices=CI)
        # real_data0 = Transformer.transform(real_data0)
        real_data1 = load_batch(dataset['real1'], n_samples_1, var_indices_real=VI, crop_indices=CI)
        # real_data1 = Transformer.transform(real_data1)
        real_data, fake_data = real_data0, real_data1

    else:
        raise ValueError(f"Dataset keys must be either 'real'/'fake' or 'real0'/'real1', not {list(dataset.keys())}")

    # the interesting part : computing each metric of metrics_list

    results = {}
    for metric in metrics_list:
        print(f"Computing metric: {metric}")
        Metric = getattr(metrics, metric)
        results[metric] = Metric(real_data, fake_data, select=False)
        print(f"\n\nResults for step {step}, for metric {metric}: {results[metric]}\n\n")
    return results, step

# def eval_distance_metrics(data, option='from_names', mean_pert=False):
#     """

#     this function should test distance metrics for datasets=[(filelist1, filelist2), ...]
#     in order to avoid memory overhead, datasets are created and destroyed dynamically

#     Inputs :

#        data : tuple of 

#            metric : str, the name of a metric in the metrics4arome namespace

#            dataset : 
#                dict of file lists (option='from_names') /str (option 'from_matrix')

#                Identifies the file names to extract data from

#                Keys of the dataset are either : 'real', 'fake' when comparing 
#                                sets of real and generated data

#                                                 'real0', 'real1' when comparing
#                                different sets of real data

#            n_samples_0, n_samples_1 : int,int , the number of samples to draw
#                                       to compute the metric.

#                                       Note : most metrics require equal numbers

#                                       by default, 0 : real, 1 : fake

#            VI, CI : iterables of indices to select (VI) variables/channels in samples
#                                                    (CI) crop indices in maps

#            index : int, identifier to the data passed 
#                    (useful only if used in multiprocessing)


#        option : str, to choose if generated data is loaded from several files (from_names)
#                or from one big Matrix (from_matrix)
       
#        mean_pert :  wether mean/perturbation splitting is used in fake data

#        iter :  whether to perform exponential tf

#     Returns :

#         results : np.array containing the calculation of the metric

#     """
#     config, metrics_list, dataset, n_samples_0, n_samples_1, Detransf, VI, VI_f, CI, step = data
#     print(f"evaluating backend distances for step: {step}")
#     ## loading and normalizing data
#     print('Loading data') 
#     if list(dataset.keys()) == ['real', 'fake']:
#         if option == 'from_names':
#             assert(type(dataset['fake']) == list)
#             fake_data = load_batch(dataset['fake'], n_samples_1, var_indices_fake=VI_f, crop_indices=CI, option='fake', mean_pert=mean_pert)
#             print('fake data loaded')
#         if option == 'from_matrix':
#             assert (type(dataset['fake'] == str))
#             fake_data = np.load(dataset['fake'], dtype=np.float32)

#         print(f"n_samples real: {n_samples_0}")
#         print(f"VI real: {VI}")
#         print(f"VI fake: {VI_f}")
#         print(f"CI real: {CI}")

#         real_data = load_batch(dataset['real'], n_samples_0, var_indices_real=VI, var_indices_fake=VI_f, crop_indices=CI)
#         Detransf.detransform(fake_data, step)

#     elif list(dataset.keys()) == ['real0', 'real1']:
#         real_data0 = load_batch(dataset['real0'], n_samples_0, var_indices_real=VI, crop_indices=CI)
#         real_data1 = load_batch(dataset['real1'], n_samples_1, var_indices_real=VI, crop_indices=CI)
#         real_data, fake_data = real_data0, real_data1

#     else:
#         raise ValueError(f"Dataset keys must be either 'real'/'fake' or 'real0'/'real1', not {list(dataset.keys())}")

#     # the interesting part : computing each metric of metrics_list

#     results = {}
#     for metric in metrics_list:
#         print(f"Computing metric: {metric}")
#         Metric = getattr(metrics, metric)
#         results[metric] = Metric(real_data, fake_data, select=False)
#     print(f"\n\nResults for metric {metric}: {results}\n\n")
#     return results, step

def global_dataset_eval(data, option='from_names', mean_pert=False):
    """

    evaluation of metric on the DataSet (treated as a single numpy matrix)

    Inputs :

        data : iterable (tuple)of str, dict, int

            metric : str, the name of a metric in the metrics4arome namespace

            dataset :
                file list /str containing the ids of the files to get samples


            n_samples_0, n_samples_1 : int, int, the number of samples to draw
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
    config, Transformer, metrics_list, dataset, n_samples, VI, VI_f, CI, step, data_option = data
    print(f"evaluating backend standalones for step: {step}")
    if option=="from_names":
        if data_option == 'fake' :
            assert(type(dataset)==list)
            print('loading fake data')
            data = load_batch(dataset, n_samples, var_indices_fake=VI_f, crop_indices=CI, option=data_option, mean_pert=mean_pert)
            data = Transformer.detransform(data, step)
            # data[:, 0] = np.clip(data[:, 0], a_min=None, a_max=350)
            
        elif data_option == 'real':
            assert (type(dataset) == list)
            print('loading real data')
            data = load_batch(dataset, n_samples, var_indices_real=VI, crop_indices=CI, option=data_option)
            # data = Transformer.transform(data)

    if option == 'from_matrix':

        assert (type(dataset) == str)

        data = np.load(dataset, dtype=np.float32)

        if iter>0:
            Means = np.load(
                real_data_dir + mean_data_file)[VI].reshape(1, len(VI), 1, 1)
            Maxs = np.load(
                real_data_dir + max_data_file)[VI].reshape(1, len(VI), 1, 1)
            data = denormalize_and_exp(data, 0.95, Means, Maxs, iter=iter)

    results = {}
    for metric in metrics_list:
        print(f"Computing metric: {metric}")
        Metric = getattr(metrics, metric)
        results[metric] = Metric(data, select=False)
    return results, step