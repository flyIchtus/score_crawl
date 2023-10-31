import os
import sys

HOME_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HOME_DIR)

import pickle as pk
from argparse import ArgumentParser

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

parser = ArgumentParser()

parser.add_argument("set", type=int, help="Set number")
parser.add_argument("config", type=str, help="config file")
parser.add_argument("--start", type=int, default=0, help="Starting checkpoint")
parser.add_argument("--stop", type=int, default=100000, help="Stopping checkpoint")
parser.add_argument("--step", type=int, default=10000, help="Step between checkpoints")

parser.add_argument("-i", "--instances", default=[1], type=int, nargs="*", help="Instance number in Set")
args = parser.parse_args()

set_number = args.set
instance_number_list = args.instances

with open(args.config_file, 'r') as main_config_yaml:
    config = yaml.safe_load(main_config_yaml)

lat_dim = config["ensemble"]["--latent_dim"][0]
bs = config["ensemble"]["--batch_size"][0]
lr_G = config["ensemble"]["--lr_G"][0]
lr_D = config["ensemble"]["--lr_D"][0]
size = str2inttuple(config["ensemble"]["--crop_indexes"][0])[1] - str2inttuple(config["ensemble"]["--crop_indexes"][0])[0]
use_noise = config["ensemble"]["--use_noise"][0]
var_names = str2list(config["ensemble"]["--var_names"][0])
tanh_output = config["ensemble"]["--tanh_output"][0]

var_names = ["rr", "u", "v", "t2m"]
path = os.path.join("scratch", "work", "gandonb", "Exp_StyleGAN", f"Set_{set_number}", f"stylegan2_stylegan_dom_{size}_lat-dim_{lat_dim}_bs_{bs}_{lr_D}_{lr_G}_ch-mul_2_vars_{'_'.join(str(var_name) for var_name in var_names)}_noise_{use_noise}")
outpath = os.path.join("scratch", "work", "gandonb", "Presentable", "plot_metric")

prefix = f"METR_clip"
# prefix = f"AROME_PHY"
arome = False
# arome = True

standalone_metrics_list = ["spectral_compute", "ls_metric", "quant_map"]
# standalone_metrics_list = ["spectral_compute"]
distance_metrics_list = ["W1_random_NUMPY", "W1_Center_NUMPY", "SWD_metric_torch"]

n_samples = 4096

if arome:
    steps_list = [0]
else:
    steps_list = [k for k in range(args.start, args.stop + 1, args.step)]
# steps_list = [0]

standalone_metrics_str = '_'.join(standalone_metrics_list)
distance_metrics_str = '_'.join(distance_metrics_list)

steps_str = "_".join(map(str, steps_list))

if arome:
    filename_standalone = f"{prefix}_{standalone_metrics_str}_1_standalone_metrics_rr_u_v_t2m_dom_256{f'_{n_samples}' * 10}.p"
    filename_distance = f"{prefix}_{distance_metrics_str}_1_distance_metrics_rr_u_v_t2m_dom_256{f'_{n_samples}' * 10}.p"
else:
    filename_standalone = f"{prefix}_{standalone_metrics_str}_1_standalone_metrics_step_{steps_str}_{n_samples}.p"
    filename_distance = f"{prefix}_{distance_metrics_str}_1_distance_metrics_step_{steps_str}_{n_samples}.p"


def quantile_map(set_number, instance_entry_list, path, outpath, filename_standalone, arome, args):
    # Define the quantile values and their corresponding vmin, vmax ranges
    v_dict = {
        0.01: {"rr": (-0.1, 0.1), "u": (-20, 3), "v": (-20, 3), "t2m": (250, 285)},
        0.1: {"rr": (-0.1, 0.5), "u": (-10, 3), "v": (-15, 3), "t2m": (255, 295)},
        0.9: {"rr": (-0.1, 0.5), "u": (0, 15), "v": (0, 10), "t2m": (275, 305)},
        0.99: {"rr": (0, 10), "u": (0, 20), "v": (0, 15), "t2m": (280, 310)},
    }
    
    for instance_entry in instance_entry_list:
        instance_path = os.path.join(path, instance_entry.name, 'log', filename_standalone)
        
        # Load the standalone_dict from the pickle file
        with open(instance_path, 'rb') as pkfile:
            standalone_dict = pk.load(pkfile)

        for quantile_idx, quantile in enumerate(sorted(v_dict.keys())):
            if arome:
                outpath_base = os.path.join(outpath, "AROME", instance_entry.name, f'quant_map/quantile_{quantile}/')
            else:
                outpath_base = os.path.join(outpath, f'Set_{set_number}', instance_entry.name, f'quant_map/quantile_{quantile}/')
            # outpath_base = os.path.join(outpath, 'AROME', instance_entry.name, f'quant_map/quantile_{quantile}/')
            outpaths_quantile_map_list = [outpath_base, os.path.join(outpath_base, 'fix/')]
            
            # Create directories if they don't exist
            for outpath_quantile_map in outpaths_quantile_map_list:
                os.makedirs(outpath_quantile_map, exist_ok=True)
            
            for idx_step, step in enumerate(steps_list):
                for idx_var, var in enumerate(var_names):
                    quantile_map = standalone_dict[0][step]["quant_map"][quantile_idx][idx_var]
                    for idx_output_path, outpath_quantile_map in enumerate(outpaths_quantile_map_list):
                        plt.clf()
                        fig, axes = plt.subplots()
                        if idx_output_path == 0:
                            img = axes.imshow(quantile_map, origin="lower")
                        else:
                            img = axes.imshow(quantile_map, origin="lower", vmin=v_dict[quantile][var][0], vmax=v_dict[quantile][var][1])
                        fig.colorbar(img)
                        fig.savefig(os.path.join(outpath_quantile_map, f'quantile_{quantile}_step_{step}_{var}.png'))
                        plt.close(fig)
                        np.save(os.path.join(outpath_quantile_map, f'quantile_{quantile}_step_{step}_{var}.npy'), quantile_map)

def W1_Center_NUMPY_plot(set_number, instance_entry_list, path, outpath, filename_distance, arome, args):
    for instance_entry in instance_entry_list:
        instance_path = os.path.join(path, instance_entry.name, 'log', filename_distance)
        
        # Load the distance_dict from the pickle file
        with open(instance_path, 'rb') as pkfile:
            distance_dict = pk.load(pkfile)

        if arome:
            outpath_base = os.path.join(outpath, "AROME", instance_entry.name, f'W1_Center')
        else:
            outpath_base = os.path.join(outpath, f'Set_{set_number}', instance_entry.name, f'W1_Center')
        # outpath_base = os.path.join(outpath, 'AROME', instance_entry.name, f'W1_Center/')
        outpaths_W1_Center_list = [outpath_base, os.path.join(outpath_base, 'fix')]
        # Create directories if they don't exist
        for outpath_W1_Center in outpaths_W1_Center_list:
            os.makedirs(outpath_W1_Center, exist_ok=True)
        for idx_output_path, outpath_W1_Center in enumerate(outpaths_W1_Center_list):
            plt.clf()
            for exp in range(len(distance_dict.keys())):
                if arome:
                    W1_Center_list = [distance_dict[exp]["W1_Center_NUMPY"][0] for step in steps_list]
                    plt.plot(steps_list, W1_Center_list, marker="o", markersize=2, label=f"Programm: {exp}")
                else:
                    W1_Center_list = [distance_dict[exp][step]["W1_Center_NUMPY"][0] for step in steps_list]
                    plt.plot(steps_list[1:], W1_Center_list[1:], marker="o", markersize=2, label=f"Programm: {exp}")
                if idx_output_path == 1:
                    plt.ylim([0, 1000])
                plt.title(f"W1 Center")
                plt.ylabel("W1 Center")
                plt.xlabel("Steps")
                plt.legend()
            plt.savefig(os.path.join(outpath_W1_Center, f'W1_Center.png'))
            plt.close()
            np.save(os.path.join(outpath_W1_Center, f'W1_Center.npy'), np.array([steps_list, W1_Center_list]))

def W1_random_NUMPY_plot(set_number, instance_entry_list, path, outpath, filename_distance, arome, args):
    for instance_entry in instance_entry_list:
        instance_path = os.path.join(path, instance_entry.name, 'log', filename_distance)
        
        # Load the distance_dict from the pickle file
        with open(instance_path, 'rb') as pkfile:
            distance_dict = pk.load(pkfile)

        if arome:
            outpath_base = os.path.join(outpath, "AROME", instance_entry.name, 'W1_random')
        else:
            outpath_base = os.path.join(outpath, f'Set_{set_number}', instance_entry.name, 'W1_random')
        outpaths_W1_random_list = [outpath_base, os.path.join(outpath_base, 'fix')]
        # Create directories if they don't exist
        for outpath_W1_random in outpaths_W1_random_list:
            os.makedirs(outpath_W1_random, exist_ok=True)
        plt.clf()
        for idx_output_path, outpath_W1_random in enumerate(outpaths_W1_random_list):
            for exp in range(len(distance_dict.keys())):
                if arome:
                    W1_random_list = [distance_dict[exp]["W1_random_NUMPY"][0] for step in steps_list]
                    plt.plot(steps_list, W1_random_list, marker="o", markersize=2, label=f"Programm: {exp}")
                else:
                    W1_random_list = [distance_dict[exp][step]["W1_random_NUMPY"][0] for step in steps_list]
                    plt.plot(steps_list[1:], W1_random_list[1:], marker="o", markersize=2, label=f"Programm: {exp}")
                if idx_output_path == 1:
                    plt.ylim([0, 1000])
                plt.title(f"W1 random")
                plt.ylabel("W1 random")
                plt.xlabel("Steps")
                plt.legend()
            plt.savefig(os.path.join(outpath_W1_random, f'W1_random.png'))
            plt.close()
            np.save(os.path.join(outpath_W1_random, f'W1_random.npy'), np.array([steps_list, W1_random_list]))


def ls_metric_plot(set_number, instance_entry_list, path, outpath, filename_standalone, arome, args):
    v_dict = {
        "rr": (0, 50),
        "u": (0, 30),
        "v": (0, 30),
        "t2m": (0, 100)
    }
    for instance_entry in instance_entry_list:
        instance_path = os.path.join(path, instance_entry.name, 'log', filename_standalone)
        
        # Load the standalone_dict from the pickle file
        with open(instance_path, 'rb') as pkfile:
            standalone_dict = pk.load(pkfile)
        if arome:
            outpath_base = os.path.join(outpath, "AROME", instance_entry.name, f'ls_metric')
        else:
            outpath_base = os.path.join(outpath, f'Set_{set_number}', instance_entry.name, f'ls_metric')
        # outpath_base = os.path.join(outpath, 'AROME', instance_entry.name, f'ls_metric/')
        outpaths_ls_metric_list = [outpath_base, os.path.join(outpath_base, 'fix')]
        
        # Create directories if they don't exist
        for outpath_ls_metric in outpaths_ls_metric_list:
            os.makedirs(outpath_ls_metric, exist_ok=True)
        for idx_step, step in enumerate(steps_list):
            for idx_var, var in enumerate(var_names):
                ls_metric_map = standalone_dict[0][step]["ls_metric"][idx_var]
                for idx_output_path, outpath_ls_metric in enumerate(outpaths_ls_metric_list):
                    save_dir = os.path.join(outpath_ls_metric, var)
                    os.makedirs(save_dir, exist_ok=True)
                    plt.clf()
                    fig, axes = plt.subplots()
                    if idx_output_path == 0:
                        img = axes.imshow(ls_metric_map, origin="lower")
                    else:
                        img = axes.imshow(ls_metric_map, origin="lower", vmin=v_dict[var][0], vmax=v_dict[var][1])
                    fig.colorbar(img)
                    fig.suptitle(f"lenght scale for {var}")
                    fig.savefig(os.path.join(save_dir, f'ls_metric_step_{step}_{var}.png'))
                    plt.close(fig)
                    np.save(os.path.join(save_dir, f'ls_metric_step_{step}_{var}.npy'), ls_metric_map)


def spectral_compute_plot(set_number, instance_entry_list, path, outpath, filename_standalone, arome, args):
    v_dict = {
        "rr": (0, 10),
        "u": (0, 30),
        "v": (0, 30),
        "t2m": (0, 100)
    }
    for instance_entry in instance_entry_list:
        instance_path = os.path.join(path, instance_entry.name, 'log', filename_standalone)
        
        # Load the standalone_dict from the pickle file
        with open(instance_path, 'rb') as pkfile:
            standalone_dict = pk.load(pkfile)
        if arome:
            outpath_base = os.path.join(outpath, "AROME", instance_entry.name, f'spectral_compute')
        else:
            outpath_base = os.path.join(outpath, f'Set_{set_number}', instance_entry.name, f'spectral_compute')
        # outpath_base = os.path.join(outpath, 'AROME', instance_entry.name, f'spectral_compute')
        outpaths_spectral_compute_list = [outpath_base, os.path.join(outpath_base, 'fix')]
        
        # Create directories if they don't exist
        for outpath_spectral_compute in outpaths_spectral_compute_list:
            os.makedirs(outpath_spectral_compute, exist_ok=True)
        for idx_step, step in enumerate(steps_list):
            for idx_var, var in enumerate(var_names):
                spectral_compute_list = standalone_dict[0][step]["spectral_compute"][idx_var]
                for idx_output_path, outpath_spectral_compute in enumerate(outpaths_spectral_compute_list):
                    save_dir = os.path.join(outpath_spectral_compute, var)
                    os.makedirs(save_dir, exist_ok=True)
                    plt.clf()
                    scale = np.linspace(2 * np.pi / 2.6, 45 * 256 // 128 * 2 * np.pi / 2.6, 45 * 256 // 128)
                    plt.plot(scale, spectral_compute_list, marker=".", label=f"Programm: 0")
                    if idx_output_path == 1:
                        plt.ylim([10e-8, 10e-2])
                    plt.title(f"Spectral compute for {var}")
                    plt.ylabel(f"Spectral compute value for {var}")
                    plt.xlabel("Scale")
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.legend()
                    plt.savefig(os.path.join(save_dir, f'spectral_compute_{var}_step_{step}.png'))
                    plt.close()
                    np.save(os.path.join(save_dir, f'spectral_compute_{var}_step_{step}.npy'), np.array([scale, spectral_compute_list]))


def SWD_torch_plot(set_number, instance_entry_list, path, outpath, filename_distance, arome, args):
    dict_scale = {
        0: 128,
        1: 64,
        2: 32,
        3: 16,
        4: "mean",
    }
    for instance_entry in instance_entry_list:
        instance_path = os.path.join(path, instance_entry.name, 'log', filename_distance)
        
        # Load the distance_dict from the pickle file
        with open(instance_path, 'rb') as pkfile:
            distance_dict = pk.load(pkfile)

        if arome:
            outpath_base = os.path.join(outpath, "AROME", instance_entry.name, f'SWD_torch')
        else:
            outpath_base = os.path.join(outpath, f'Set_{set_number}', instance_entry.name, f'SWD_torch')
        # outpath_base = os.path.join(outpath, 'AROME', instance_entry.name, f'SWD_torch/')
        outpaths_SWD_torch_list = [outpath_base, os.path.join(outpath_base, 'fix')]
        # Create directories if they don't exist
        for outpath_SWD_torch in outpaths_SWD_torch_list:
            os.makedirs(outpath_SWD_torch, exist_ok=True)
        for idx_output_path, outpath_SWD_torch in enumerate(outpaths_SWD_torch_list):
            if arome:
                SWD_elem_num = len(distance_dict[0]["SWD_metric_torch"].numpy())
            else:
                SWD_elem_num = len(distance_dict[0][steps_list[0]]["SWD_metric_torch"].numpy())
            for scale_idx in range(SWD_elem_num):
                plt.clf()
                for exp in range(len(distance_dict.keys())):
                    if arome:
                        SWD_torch_list = [distance_dict[exp]["SWD_metric_torch"].numpy()[scale_idx] for step in steps_list]
                    else:
                        SWD_torch_list = [distance_dict[exp][step]["SWD_metric_torch"].numpy()[scale_idx] for step in steps_list]
                    plt.plot(steps_list[1:], SWD_torch_list[1:], marker=".", label=f"Programm: {exp}")
                    if idx_output_path == 1:
                        plt.ylim([0, 1000])
                    plt.title(f"SWD_{dict_scale[scale_idx]}")
                    plt.ylabel(f"SWD_{dict_scale[scale_idx]}")
                    plt.xlabel("Steps")
                    plt.legend()
                plt.savefig(os.path.join(outpath_SWD_torch, f'SWD_{dict_scale[scale_idx]}.png'))
                plt.close()
                np.save(os.path.join(outpath_SWD_torch, f'SWD_{dict_scale[scale_idx]}.npy'), np.array([steps_list, SWD_torch_list]))


def one_set(set_number, path, outpath, filename_distance, filename_standalone, arome, args):
    instance_entry_list = sorted([instance for instance in os.scandir(path)], key=lambda instance: int(instance.name[9:]))
    print("Plotting quantile maps...")
    quantile_map(set_number, instance_entry_list, path, outpath, filename_standalone, arome, args)
    print("Plotting ls metric maps...")
    ls_metric_plot(set_number, instance_entry_list, path, outpath, filename_standalone, arome, args)
    print("Plotting spectra...")
    spectral_compute_plot(set_number, instance_entry_list, path, outpath, filename_standalone, arome, args)
    print("Plotting W1 center...")
    W1_Center_NUMPY_plot(set_number, instance_entry_list, path, outpath, filename_distance, arome, args)
    print("Plotting W1 random...")
    W1_random_NUMPY_plot(set_number, instance_entry_list, path, outpath, filename_distance, arome, args)
    print("Plotting SWD...")
    SWD_torch_plot(set_number, instance_entry_list, path, outpath, filename_distance, arome, args)


one_set(set_number, path, outpath, filename_distance, filename_standalone, arome, args)