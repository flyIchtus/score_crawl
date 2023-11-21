import concurrent.futures
import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from collections import Counter
from multiprocessing.pool import ThreadPool as Pool
from time import perf_counter

import matplotlib
import matplotlib.colors as colors
import seaborn as sns
import yaml

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from called.utile import (make_save_dir, parse_float, print_progress,
                          print_progress_bar)
from matplotlib import pyplot as plt

THRESHOLD_LIST = [0, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30]# , 35, 50, 55, 60, 70, 80, 90, 100, 110, 135, 150, 170, 200, 250, 300, 350]
class Instance:
    def __init__(self, set_number, step, gridshape):
        self.orders_ = 3
        self.set_number_ = set_number
        self.step_ = step
        self.gridsize_ = gridshape[0] * gridshape[1]
        self.stats_arrays_count_ = {"extracted_values": Counter(), "area_proportion": {k: 0 for k in THRESHOLD_LIST}}
        self.orders_maps_list_ = [np.zeros(gridshape) for _ in range(self.orders_)]
        self.stats_numbers_dict_ = {"global_rate": 0, "mean": 0, "n_files": 0, "n_files_no_rain": 0, "n_files_greater_than_1mm": 0, "n_files_greater_than_5mm": 0}

    def reset(self, gridshape):
        self.stats_arrays_count_ = {"extracted_values": Counter(), "area_proportion": {k: 0 for k in THRESHOLD_LIST}}
        self.orders_maps_list_ = [np.zeros(gridshape) for _ in range(self.orders_)]
        self.stats_numbers_dict_ = {"global_rate": 0, "mean": 0, "n_files": 0, "n_files_no_rain": 0, "n_files_greater_than_1mm": 0, "n_files_greater_than_5mm": 0}

    def save_order_maps(self, save_dir):
        for order, order_map in enumerate(self.orders_maps_list_):
            np.save(os.path.join(save_dir, f"order_{order}.npy"), order_map)

    def save_counts(self, save_dir):
        for name, array in self.stats_arrays_count_.items():
            with open(os.path.join(save_dir, f"{name}.json"), "w", encoding="utf8") as countfile:
                json_count = json.dumps(array, indent=4)
                countfile.write(json_count)

    def save_plots(self, save_dir_img):
        for order, order_map in enumerate(self.orders_maps_list_):
            plt.clf()
            fig, axes = plt.subplots()
            img = axes.imshow(order_map, origin="lower")
            fig.colorbar(img)
            fig.savefig(os.path.join(save_dir_img, f"order_{order}.png"))
            plt.close(fig)
        plt.clf()
        keys = np.array(sorted(list(map(float, self.stats_arrays_count_["area_proportion"].keys()))))
        values = np.array([self.stats_arrays_count_["area_proportion"][key] for key in keys])
        plt.plot(keys, values)
        plt.yscale("log")
        plt.xlabel("s_rr")
        plt.ylabel("Area proportion (log10)")
        plt.title("Area proportion for precipitation >= s_rr")
        plt.savefig(os.path.join(save_dir_img, "area_proportion.png"))
        print(save_dir_img)
        if len(self.stats_arrays_count_["extracted_values"]):
            keys = sorted(list(map(float, self.stats_arrays_count_["extracted_values"].keys())))
            values = [self.stats_arrays_count_["extracted_values"][str(key)] for key in keys]
            v_couple_list = [(0, 20), (1, 15), (4, 20), (20, 70)]
            for v_couple in v_couple_list:
                v_min, v_max = v_couple
                self.sns_plot(keys, values, v_min, v_max, save_dir_img)
    
    def sns_plot(self, keys, values, v_min, v_max, save_dir_img):
        plt.clf()
        plt.hist(x=keys, weights=values, density=True, bins=(v_max-v_min) * 2, range=(v_min, v_max))
        plt.xlabel("rr")
        plt.savefig(os.path.join(save_dir_img, f"extracted_values_{v_min}_{v_max}.png"))

    def save_patch(self, gigafile_name, verbose_level, args):
        if args.verbose >= verbose_level:
            print(f"{os.getpid()} -> Saving stats for {self.set_number_}")
        save_dir = os.path.join(save_path, f"~Set_{self.set_number_}", f"step_{self.step_}", "~stats", f"{gigafile_name[:-4]}")
        make_save_dir(save_dir, args)

        self.save_order_maps(save_dir)
        self.save_counts(save_dir)

        json_numbers = json.dumps(self.stats_numbers_dict_, indent=4)
        with open(os.path.join(save_dir, "log.json"), "w", encoding="utf8") as logfile:
            logfile.write(json_numbers)
    
    def compute_numbers(self):
        self.stats_numbers_dict_["global_rate"] = np.mean(self.orders_maps_list_[0])
        self.stats_numbers_dict_["mean"] = np.mean(self.orders_maps_list_[1])

    def save(self, verbose_level, args):
        if args.verbose >= verbose_level:
            print(f"Saving stats for {self.set_number_}")
        save_dir = os.path.join(save_path, f"~Set_{self.set_number_}", f"step_{self.step_}", "~stats")
        save_dir_img = os.path.join(save_dir, "pictures")
        make_save_dir(save_dir_img, args)

        self.save_order_maps(save_dir)
        self.save_counts(save_dir)
        self.save_plots(save_dir_img)

        json_numbers = json.dumps(self.stats_numbers_dict_, indent=4)
        with open(os.path.join(save_dir, "log.json"), "w", encoding="utf8") as logfile:
            logfile.write(json_numbers)

    def update_stats(self, data_path, gigafile):
        data_dir = os.path.join(data_path, gigafile.name[:-4])
        with open(os.path.join(data_dir, "log.json")) as logfile:
            stats_number_dict = json.load(logfile)
            for name in ["n_files", "n_files_no_rain", "n_files_greater_than_1mm", "n_files_greater_than_5mm"]:
                self.stats_numbers_dict_[name] += stats_number_dict[name]
          
    def update_order_map(self, gigafile_set, data_path, order, order_map):
        for gigafile in gigafile_set:
            data_dir = os.path.join(data_path, gigafile.name[:-4])
            order_map_patch = np.load(os.path.join(data_dir, f"order_{order}.npy"))
            order_map += order_map_patch
        order_map /= self.stats_numbers_dict_["n_files"]
        self.orders_maps_list_[order] = order_map
    
    def update_stats_arrays(self, data_path, gigafile_set):
        for name in self.stats_arrays_count_:
            for gigafile in gigafile_set:
                data_dir = os.path.join(data_path, gigafile.name[:-4])
                with open(os.path.join(data_dir, f"{name}.json")) as countfile:
                    self.stats_arrays_count_[name].update(json.load(countfile))
        self.stats_arrays_count_["area_proportion"] = {float(key): value / (self.stats_numbers_dict_["n_files"] * self.gridsize_) for key, value in self.stats_arrays_count_["area_proportion"].items()}

    def cleanup_gigafiles_stats(self):
        for gigafile_stats in os.scandir(os.path.join(save_path, f"~Set_{self.set_number_}", f"step_{self.step_}", "~stats")):
            shutil.rmtree(gigafile_stats.path)


class Parameter:
    def __init__(self, set_number, step, gridshape):
        self.orders_ = 3
        self.set_number_ = set_number
        self.step_ = step
        self.gridsize_ = gridshape[0] * gridshape[1]
        self.stats_arrays_count_ = {"extracted_values": Counter(), "area_proportion": Counter()}
        self.orders_maps_list_ = [np.zeros(gridshape) for _ in range(self.orders_)]
        self.stats_numbers_dict_ = {"global_rate": 0, "mean": 0, "n_files": 0, "n_files_no_rain": 0, "n_files_greater_than_1mm": 0, "n_files_greater_than_5mm": 0, "ratio_n_files_no_rain_over_n_files_total": 0, "ratio_n_files_greater_than_1mm_over_n_files_total": 0, "ratio_n_files_greater_than_5mm_over_n_files_total": 0}

    def save_order_maps(self, save_dir):
        for order, order_map in enumerate(self.orders_maps_list_):
            np.save(os.path.join(save_dir, f"order_{order}.npy"), order_map)

    def save_counts(self, save_dir):
        for name, array in self.stats_arrays_count_.items():
            with open(os.path.join(save_dir, f"{name}.json"), "w", encoding="utf8") as countfile:
                json_count = json.dumps(array, indent=4)
                countfile.write(json_count)

    def save_plots(self, save_dir_img):
        for order, order_map in enumerate(self.orders_maps_list_):
            plt.clf()
            fig, axes = plt.subplots()
            img = axes.imshow(order_map, origin="lower")
            fig.colorbar(img)
            fig.savefig(os.path.join(save_dir_img, f"order_{order}.png"))
            plt.close(fig)
        plt.clf()
        keys = np.array(sorted(list(map(float, self.stats_arrays_count_["area_proportion"].keys()))))
        values = np.array([self.stats_arrays_count_["area_proportion"][str(key)] for key in keys])
        plt.plot(keys, values)
        plt.yscale("log")
        plt.xlabel("s_rr")
        plt.ylabel("Area proportion (log10)")
        plt.title("Area proportion for precipitation >= s_rr")
        plt.savefig(os.path.join(save_dir_img, f"area_proportion.png"))
        if len(self.stats_arrays_count_["extracted_values"]):
            keys = sorted(list(map(float, self.stats_arrays_count_["extracted_values"].keys())))
            values = [self.stats_arrays_count_["extracted_values"][str(key)] for key in keys]
            v_couple_list = [(0, 20), (1, 15), (4, 20), (20, 70)]
            for v_couple in v_couple_list:
                v_min, v_max = v_couple
                # try:
                #     v_max = keys.index(v_max)
                self.sns_plot(keys, values, v_min, v_max, save_dir_img)
                # except ValueError:
                #     print(f"{v_max} is too high, no plot for {v_couple}.")
    
    def sns_plot(self, keys, values, v_min, v_max, save_dir_img):
        plt.clf()
        plt.hist(x=keys, weights=values, density=True, bins=(v_max-v_min) * 2, range=(v_min, v_max))
        plt.savefig(os.path.join(save_dir_img, f"extracted_values_{v_min}_{v_max}.png"))

    def save(self, verbose_level, args):
        if args.verbose >= verbose_level:
            print(f"Saving stats global")
        save_dir = os.path.join(save_path, f"~Set_{self.set_number_}", f"step_{self.step_}", "~stats")
        save_dir_img = os.path.join(save_dir, "pictures")
        make_save_dir(save_dir_img, args)

        self.save_order_maps(save_dir)
        self.save_counts(save_dir)
        self.save_plots(save_dir_img)
        json_numbers = json.dumps(self.stats_numbers_dict_, indent=4)
        with open(os.path.join(save_dir, "log.json"), "w", encoding="utf8") as logfile:
            logfile.write(json_numbers)
    
    def update(self, instance):
        data_dir = os.path.join(save_path, f"~Set_{instance.set_number_}", f"step_{instance.step_}", "~stats")
        self.update_stats(data_dir)
        for order, order_map in enumerate(self.orders_maps_list_):
            self.update_order_map(instance, order, order_map)
        self.update_stats_arrays(instance)
        self.divide()

    def update_stats(self, data_dir):
        with open(os.path.join(data_dir, "log.json")) as logfile:
            stats_number_dict = json.load(logfile)
            exclusion_list = ["ratio_n_files_no_rain_over_n_files_total", "ratio_n_files_greater_than_1mm_over_n_files_total", "ratio_n_files_greater_than_5mm_over_n_files_total"]
            for name in self.stats_numbers_dict_:
                if name not in exclusion_list:
                    self.stats_numbers_dict_[name] += stats_number_dict[name]

    def update_order_map(self, instance, order, order_map):
        data_dir = os.path.join(save_path, f"~Set_{instance.set_number_}", f"step_{instance.step_}", "~stats")
        order_map_patch = np.load(os.path.join(data_dir, f"order_{order}.npy"))
        order_map += order_map_patch
        self.orders_maps_list_[order] = order_map
    
    def update_stats_arrays(self, instance):
        for name in self.stats_arrays_count_:
            data_dir = os.path.join(save_path, f"~Set_{instance.set_number_}", f"step_{instance.step_}", "~stats")
            with open(os.path.join(data_dir, f"{name}.json")) as countfile:
                self.stats_arrays_count_[name].update(json.load(countfile))
    

    def divide(self):
        if self.stats_numbers_dict_["n_files"]:
            for name in ["n_files_no_rain", "n_files_greater_than_1mm", "n_files_greater_than_5mm"]:
                self.stats_numbers_dict_[f"ratio_{name}_over_n_files_total"] = self.stats_numbers_dict_[name] / self.stats_numbers_dict_["n_files"]

def run_stat(_Fsample_dir, set_number, step, variable, config, gridshape, args):
    giga_dir = _Fsample_dir
    gigafiles_set = {gigafile for gigafile in os.scandir(giga_dir) if (f"_Fsample_{step}_" in gigafile.name and "~" not in gigafile.name)}
    gigafiles_set = sorted(gigafiles_set, key=lambda gigafile_entry: int(gigafile_entry.name[10 + len(str(step)):-4]))
    n_gigafiles = len(gigafiles_set)
    Transformer = Transform(config, gridshape)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_gigafile, gigafile.name, gigafile.path, set_number, step, variable, Transformer, gridshape, args) for idx_gigafile, gigafile in enumerate(gigafiles_set)}

    for idx, future in enumerate(concurrent.futures.as_completed(future_to_idx)):
        future.result()
        if args.verbose >= 4:
            print(f"Gigafile {idx + 1}/{n_gigafiles} processed successfully.")
        # except Exception as exc:
        #     print(f"An error occurred while processing gigafile {idx + 1}/{n_gigafiles}: {exc}")

    print("Processing all gigafiles completed.")
    executor.shutdown(wait=True)

    instance = Instance(set_number, step, gridshape)

    compute_stats_over_all_patches(instance, gigafiles_set, args)
    data_dir = os.path.join(save_path, f"~Set_{instance.set_number_}", f"step_{instance.step_}", "~stats")
    global_stats(instance, data_dir, gridshape, args)

def global_stats(instance, data_dir, gridshape, args):
    parameter = Parameter(instance.set_number_, instance.step_, gridshape)
    parameter.update(instance)
    parameter.save(3, args)


def process_gigafile(gigafile_name, gigafile_path, set_number, step, variable, Transformer, gridshape, args):
    instance = initialize_instance(set_number, step, gridshape)

    print(f"{os.getpid()} -> Loading {gigafile_name}")
    l_grid = np.load(gigafile_path)

    l_grid = Transformer.detransform(l_grid, step)
    l_grid[:, 0] = np.clip(l_grid[:, 0], a_min=None, a_max=350)
    process_instance(instance, gigafile_name, l_grid, variable, gridshape, args)
    instance.save_patch(gigafile_name, 4, args)
    instance.reset(gridshape)

def denormalize_and_exp(BigMat, scale, Mean, Max, iter = 1):
    """

    De-normalize [ = set to physical scale] samples with specific Mean and max + rescaling
    Then perform iterations of exponential on the data
    Inputs :

        BigMat : ndarray, samples to rescale

        scale : float, scale to set maximum amplitude of samples

        Mean, Max : ndarrays, must be broadcastable to BigMat

        iter :  nb of exponential iterations to be applied; 1 -> exp(data - 1.0), 2 -> exp(exp(data-1)-1)

    Returns :

        res : ndarray, same dimensions as BigMat

    """
    
    res = (1/ scale) * Max * BigMat + Mean 
    
    for j in range(iter):

        res = np.exp(res) - 1.0
    return res

def initialize_instance(set_number, step, gridshape):
    return Instance(set_number, step, gridshape)


def process_instance(instance, gigafile_name, l_grid, variable, gridshape, args):
    instance.stats_arrays_count_["extracted_values"].update(extract_values_greater_than_threshold(l_grid, variable, args))
    instance.stats_arrays_count_["area_proportion"].update(count_pixels_greater_than(variable, l_grid, args))

    instance.orders_maps_list_ = [order_map for order_map in sum_map_values_greater_than_threshold(l_grid, variable, instance.orders_, gridshape, args)]
    instance.stats_numbers_dict_["n_files"] = len(l_grid)
    for grid in l_grid:
        max_rr = np.max(grid[variable])
        if max_rr <= 0.01:
            instance.stats_numbers_dict_["n_files_no_rain"] += 1
        if max_rr >= 1:
            instance.stats_numbers_dict_["n_files_greater_than_1mm"] += 1
            if max_rr >= 5:
                instance.stats_numbers_dict_["n_files_greater_than_5mm"] += 1

def compute_stats_over_all_patches(instance, gigafile_set, args):
    data_path = os.path.join(save_path, f"~Set_{instance.set_number_}", f"step_{instance.step_}", "~stats")
    for gigafile in gigafile_set:
        instance.update_stats(data_path, gigafile)

    for order, order_map in enumerate(instance.orders_maps_list_):
        instance.update_order_map(gigafile_set, data_path, order, order_map)

    instance.update_stats_arrays(data_path, gigafile_set)

    instance.cleanup_gigafiles_stats()
    instance.compute_numbers()
    instance.save(3, args)

def sum_map_values_greater_than_threshold(data, variable, power_max, gridshape, args):
    if args.verbose >= 6:
        print(f"\nComputing sum_map for values greater than {args.threshold}...")

    args_list = [(data, variable, power, gridshape, args) for power in range(power_max)]

    with Pool(min(power_max, os.cpu_count() // 4)) as pool:
        sum_map = pool.map(sum_map_parallel, args_list, chunksize=1)  # Adjust chunksize as needed

    return sum_map

def sum_map_parallel(args):
    data, variable, power, gridshape, args = args
    n_grid = len(data)
    sum_map = np.zeros(gridshape)
    for idx, grid in enumerate(data):
        if args.verbose >= 6 and n_grid > 10:  # Print progress only if there are enough iterations
            print_progress_bar(idx, n_grid)
        sum_map += ((grid[variable] ** power) * (grid[variable] > args.threshold))
    return sum_map

def extract_values_greater_than_threshold(data, variable, args):
    """Extract from each grid all the values greater than threshold and store them in a list

    Args:
        data (list): list of the loaded data
        variable (int): index of the channels corresponding to the variable
        args (argparse.Namespace): args of the program

    Returns:
        list[float]: store every value greater than the threshold
    """
    if args.verbose >= 6:
        print(f"\nExtracting values greater than {args.threshold}...")

    n_grid = len(data)
    args_list = [(grid, variable, args) for grid in data]

    with Pool(min(2, n_grid)) as pool:
        extracted_values_list = pool.map(extract_parallel, args_list, chunksize=8000)
    
    extracted_values_count = Counter()
    for extracted_values in extracted_values_list:
        extracted_values_count.update(extracted_values)

    return extracted_values_count

def extract_parallel(args_list):
    grid, variable, args = args_list
    mask = grid[variable] > args.threshold
    extract = np.round(2 * grid[variable][mask]) / 2
    extract = [float(elem) for elem in extract]
    return Counter(extract)
    
def count_pixels_greater_than(variable, grids, args):
    counter_dict = {k: 0 for k in THRESHOLD_LIST}
    for idx_threshold, threshold in enumerate(THRESHOLD_LIST):
        counter_dict[threshold] += int(np.sum(grids[:, 0, :, :] > threshold))
    return counter_dict

def compute_area_greater_than(variable, data_dir, gridshape, args):
    """Extract from each grid all the values greater than threshold and store them in a list

    Args:
        variable (int): index of the channels corresponding to the variable
        data (list): list of the loaded data
        args (argparse.Namespace): args of the program

    Returns:
        list[float]: store every value greater than the threshold
    """
    threshold_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30]
    gigafile_set = {gigafile for gigafile in os.scandir(data_dir) if gigafile.name not in ["area_proportions.npy", "area_proportions.png", "labels.csv"]}
    n_gigafiles = len(gigafile_set)
    x_length, y_length = gridshape[0], gridshape[1]
    l_mean_proportion = np.zeros([len(threshold_list)])
    start_time = perf_counter()
    n_grid = 0
    for idx_gigafile, gigafile in enumerate(gigafile_set):
        if (idx_gigafile + 1) % ((n_gigafiles // args.refresh) + 1) == 0:
            print_progress(idx_gigafile, n_gigafiles, start_time)
        print(f"Loading gigafile {gigafile.name} ({idx_gigafile + 1}/{n_gigafiles})")
        l_grid = np.load(gigafile.path)
        n_grid += len(l_grid)
        for idx_threshold, threshold in enumerate(threshold_list):
            if args.verbose >= 4: print_progress_bar(idx_threshold, len(threshold_list))
            for idx, grid in enumerate(l_grid):
                mask = grid[variable] > threshold
                extracted_values = grid[variable][mask]
                l_mean_proportion[idx_threshold] += len(extracted_values)
    l_mean_proportion /= (gridshape[0] * gridshape[1] * n_grid)
    return l_mean_proportion

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
        elif normalization_type == "quant":
            maxs, mins = self.load_stat_files(normalization_type, "Q99", "Q01")
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
        file_path = os.path.join(data_min_max, stat_folder, mean_or_max_filename)
        means_or_maxs = np.load(file_path).astype('float32')
        print(f"{str1} file found")

        file_path = os.path.join(data_min_max, stat_folder, std_or_min_filename)
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
        elif norm_type == "minmax" or norm_type == "quant":
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
        # print("Detransform OK.")
        return data
    
    def print_data_detransf(self, data, step):
        save_dir = f"detranformed/"
        os.makedirs(save_dir, exist_ok=True)
        idd=id(data)
        plt.clf()
        fig, axes = plt.subplots()
        img = axes.imshow(data[0], origin="lower")
        fig.colorbar(img)
        fig.savefig(f"{save_dir}{idd}.png")
        print("Printing data...")
        self.online_sample_plot(data, idd)

        print("Data printed.")
    
    def online_sample_plot(self, batch, idd, mean_pert=False):
        bounds = np.array([0, 0.5, 1, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 1000])
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=18)
        cmapRR = colors.ListedColormap(["white", "#63006e", "#0000ff", "#00b2ff", "#00ffff", "#08dfd6", "#1cb8a5", "#6ba530", "#ffff00", "#ffd800", "#ffa500", "#ff0000", "#991407", "#ff00ff", "#a4ff00", "#00fa00", "#31be00", "#31858b"], name="from_list", N=None)
        batch_to_print = batch
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

        for i, var in enumerate(["rr"]):
            varname, cmap, limits = variable_mapping.get(var)

            fig, axs = plt.subplots()
            st = fig.suptitle(f"{varname}{' pert' if mean_pert else ''}", fontsize='30')
            st.set_y(0.96)
            b = batch_to_print[i]
            if var == "rr":
                im = axs.imshow(b[::1, :], origin="lower", cmap=cmap, norm=norm)
            else:
                im = axs.imshow(b[::-1, :], cmap=cmap, vmin=limits[0], vmax=limits[1])
            
            cb = fig.colorbar(im)
            cb.ax.tick_params(labelsize=20)

            save_dir = f"detranformed/"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}{idd}_ext.png")
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

def str2list(li):
    if type(li)==list:
        li2 = li
        return li2
    
    elif type(li)==str:
        li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def str2intlist(li):
    if type(li)==list:
        li2 = [int(p) for p in li]
        return li2
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 = [int(p) for p in li2]
        return li3

    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def str2inttuple(li):
    if type(li)==list:
        li2 =[int(p) for p in li]  
        return tuple(li2)
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 =[int(p) for p in li2]

        return tuple(li3)

    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

parser = ArgumentParser()

parser.add_argument("set", type=int, help="Set number")
parser.add_argument("start", type=int, help="Beginning step")
parser.add_argument("stop", type=int, help="Stopping step")
parser.add_argument("space", type=int, help="Space between step")
parser.add_argument("-i", "--instances", default=[1], type=int, nargs="*", help="Instance number in Set")

parser.add_argument("-r", "--refresh", type=int, default=25, help="Progress is shown 'refresh' times")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="Threshold for stats")

args = parser.parse_args()

config_path = os.path.join("/home", "gmap", "mrmn", "gandonb", "SAVE", "styleganPNRIA", "gan", "configs", f"Set_{args.set}")
config_file = os.path.join(config_path, "main_256.yaml")
with open(config_file, 'r') as main_config_yaml:
    config = yaml.safe_load(main_config_yaml)

lat_dim = config["ensemble"]["--latent_dim"][0]
bs = config["ensemble"]["--batch_size"][0]
lr_G = config["ensemble"]["--lr_G"][0]
lr_D = config["ensemble"]["--lr_D"][0]
size = str2inttuple(config["ensemble"]["--crop_indexes"][0])[1] - str2inttuple(config["ensemble"]["--crop_indexes"][0])[0]
use_noise = config["ensemble"]["--use_noise"][0]
var_names = str2list(config["ensemble"]["--var_names"][0])
tanh_output = config["ensemble"]["--tanh_output"][0]


max_workers = 15

set_number = args.set
dir_string = f"stylegan2_stylegan_dom_{size}_lat-dim_{lat_dim}_bs_{bs}_{lr_D}_{lr_G}_ch-mul_2_vars_{'_'.join(str(var_name) for var_name in var_names)}_noise_{use_noise}"
instance_number = 1
_Fsample_dir = os.path.join("/scratch", "work", "gandonb", "Exp_StyleGAN", f"Set_{set_number}", dir_string, f"Instance_{instance_number}", "samples")
data_min_max = config["data_dir"]
save_path = os.path.join("/scratch", "work", "gandonb", "making_stats")
config_d = Namespace()
config_d.data_transform_config_filename = os.path.join(config_path, config["ensemble"]["--dataset_handler_config"][0])
with open(config_d.data_transform_config_filename, "r") as data_transform_config_file: 
    data_transform_config_yaml = yaml.safe_load(data_transform_config_file)
data_transform_config = data_transform_config_yaml
config_d.data_transform_config = data_transform_config
gridshape = (size, size)

steps_list = [k for k in range(args.start, args.stop + 1, args.space)]

for step in steps_list:
    print(f"Step {step}")
    run_stat(_Fsample_dir, set_number, step, 0, config_d, gridshape, args)
#### STATS ON SOURCE ####
# ## PATH ##
# DATA_DIR = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/cropped_giga/"
# SAVE_DIR = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/analysis/source/"
# make_save_dir(SAVE_DIR, args)

# stat_on_source(DATA_DIR, SAVE_DIR, args)

#### COMPUTE AREA ####
## PATH ## 
# DATA_DIR = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/data_for_importance_sampling/pre_proc_11-08/cropped_120_376_540_796_giga/"
# l_thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30]
# l_mean = compute_area_greater_than(0, DATA_DIR, [256, 256], l_thresholds, args)
# print(l_mean)
# np.save(f"{DATA_DIR}area_proportions.npy", l_mean)
# l_mean = np.load(f"{DATA_DIR}area_proportions.npy")
# plt.plot(l_thresholds, l_mean)
# plt.yscale("log")
# plt.xlabel("s_rr")
# plt.ylabel("Area proportion (log10)")
# plt.title("Area proportion for precipitation >= s_rr")
# plt.savefig(f"{DATA_DIR}area_proportionslog10.png")
# plt.clf()
# plt.plot(l_thresholds, l_mean)
# plt.xlabel("s_rr")
# plt.ylabel("Area proportion (log10)")
# plt.title("Area proportion for precipitation >= s_rr")
# plt.savefig(f"{DATA_DIR}area_proportions.png")




# GIGA_DIRS = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/pre_proc_31-07-10h/cropped_giga/test/"
# DATA_DIR = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/importance_sampling/"
# DIRS = (GIGA_DIRS, DATA_DIR)
# variable = 0
# run_stat(DIRS, 0, [256, 256], args)
# compute_stats_on_every_session(DATA_DIR, CSV_DIR, variable, args)


#### VERIFY SPLIT ####
# ## PATH ##
# RAW_DATA_DIR = "/cnrm/recyf/NO_SAVE/Data/users/brochetc/float32_t2m_u_v_rr/"
# DATA_DIR = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/pre_proc_31-07-10h/cropped_giga/"
# SAVE_DIR = "/cnrm/recyf/NO_SAVE/Data/users/gandonb/importance_sampling/output/pre_proc_31-07-10h/draft/"

# GRID_NUM = 0
# VMAX = 0.02
# make_save_dir(SAVE_DIR, args)
# l_grid = np.load(DATA_DIR + str(1) + ".npy", allow_pickle=True)
# fig, axes = plt.subplots()
# img = axes.imshow(l_grid[GRID_NUM][0], origin="lower", vmin=0, vmax=VMAX)
# fig.colorbar(img)
# fig.savefig(SAVE_DIR + "rr.png")
# dataframe = pd.read_csv(DATA_DIR + "labels.csv")
# row = dataframe.iloc[GRID_NUM]
# orig_grid = np.load(RAW_DATA_DIR + row["Date"] + "_rrlt1-24.npy", allow_pickle=True)
# fig, axes = plt.subplots()
# img = axes.imshow(orig_grid[:, :, int(row["Leadtime"])-1, int(row["Member"])-1], origin="lower", vmin=0, vmax=VMAX)
# fig.colorbar(img)
# fig.savefig(SAVE_DIR + "rr_orig.png")



