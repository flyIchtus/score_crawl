import os
import pickle as pk
from argparse import ArgumentParser

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

parser = ArgumentParser()

parser.add_argument("-s", "--set", default=1, type=int, help="Set number, use as: -s=2")
parser.add_argument("-l", "--sets_list", default=None, type=int, nargs="*", help="If you want to compare several sets, use as: -l 1 2 5 6")
parser.add_argument("-n", "--num", default=1, type=int, help="Num number, use as: -n=1")
parser.add_argument("-m", "--num_list", default=None, type=int, nargs="*", help="If you want to compare several instances from one set, use as: -m 1 2 5 6")
parser.add_argument("-i", "--instances", default=[1], type=int, nargs="*", help="Instance number in Set")


args = parser.parse_args()

def make_save_dir(save_dir):
    if not os.path.exists(save_dir) and not os.path.exists(save_dir + "_done"):
        print(f"creating save dir {save_dir}")
        os.makedirs(save_dir)

set_number = args.set
sets_list = args.sets_list
num_list = args.num_list
num = args.num
instance_number_list = args.instances

path = f"/scratch/mrmn/gandonb/Exp_StyleGAN/Set_{set_number}/stylegan2_stylegan_dom_256_lat-dim_512_bs_32_0.0002_0.0002_ch-mul_2_vars_rr_noise_True/"
outpath = f"/scratch/mrmn/gandonb/Presentable/plot_metric/"
steps_list = [k * 2000 for k in range(26)]
steps_str = "_".join([str(step) for step in steps_list])
filename_distance = f"test_distance_1_distance_metrics_step_{steps_str}_16384.p"
filename_standalone = f"test_standalone_1_standalone_metrics_step_{steps_str}_16384.p"


def quantile_map(set_number, instance_entry_list, path, outpath, filename_distance, filename_standalone, args):
    for instance_entry in instance_entry_list:
        with open(f"{path}{instance_entry.name}/log/{filename_standalone}", 'rb') as pkfile:
            standalone_dict = pk.load(pkfile)
        v_dict = {0.01: (-0.1, 0.5), 0.1: (-0.1, 0.5), 0.9: (-0.1, 5), 0.99: (0, 10)}
        for quantile_idx, quantile in enumerate(sorted(v_dict.keys())):
            outpath_base = f"{outpath}Set_{set_number}/{instance_entry.name}quant_map/quantile_{quantile}/"
            outpaths_quantile_map_list = [outpath_base, f"{outpath_base}fix/"]
            for outpath_quantile_map in outpaths_quantile_map_list:
                make_save_dir(outpath_quantile_map)
            for idx_step, step in enumerate(steps_list):
                if len(steps_list) == 1:
                    quantile_map = standalone_dict["quant_map"][quantile_idx]
                else:
                    quantile_map = standalone_dict["quant_map"][idx_step][quantile_idx]
                for idx_output_path, outpath_quantile_map in enumerate(outpaths_quantile_map_list):
                    plt.clf()
                    fig, axes = plt.subplots()
                    if idx_output_path == 0:
                        img = axes.imshow(quantile_map, origin="lower")
                    else:
                        img = axes.imshow(quantile_map, origin="lower", vmin=v_dict[quantile][0], vmax=v_dict[quantile][1])
                    fig.colorbar(img)
                    fig.savefig(f"{outpath_quantile_map}step_{step}.png")
                    plt.close(fig)

def W1_Center_NUMPY_plot(set_number, instance_entry_list, path, outpath, filename_distance, filename_distance, args):
    outpath_base = f"{outpath}Set_{set_number}/W1_Center_NUMPY/"
    outpaths_W1_Center_NUMPY_list = [outpath_base, f"{outpath_base}fix/"]
    for outpath_W1_Center_NUMPY in outpaths_W1_Center_NUMPY_list:
        make_save_dir(outpath_W1_Center_NUMPY)
    for idx_output_path, outpath_W1_Center_NUMPY in enumerate(outpaths_W1_Center_NUMPY_list):
        plt.clf()
        for instance_entry in instance_entry_list:
            with open(f"{path}{instance_entry.name}/log/{filename_distance}", 'rb') as pkfile:
                distance_dict = pk.load(pkfile)
            for idx_step, step in enumerate(steps_list):
                if len(steps_list) == 1:
                    W1_Center_NUMPY = distance_dict["W1_Center_NUMPY"][quantile_idx]
                else:
                    W1_Center_NUMPY = standalone_dict["W1_Center_NUMPY"][idx_step][quantile_idx]
                plt.clf()
                fig, axes = plt.subplots()
                if idx_output_path == 0:
                    img = axes.imshow(quantile_map, origin="lower")
                else:
                    img = axes.imshow(quantile_map, origin="lower", vmin=v_dict[quantile][0], vmax=v_dict[quantile][1])
                fig.colorbar(img)
                fig.savefig(f"{outpath_quantile_map}step_{step}.png")
                plt.close(fig)



def one_set(set_number, path, outpath, filename_distance, filename_standalone, args):
    instance_entry_list = sorted([instance for instance in os.scandir(path)], key=lambda instance: int(instance.name[9:]))
    quantile_map(set_number, instance_entry_list, path, outpath, filename_distance, filename_standalone, args)


one_set(set_number, path, outpath, filename_distance, filename_standalone, args)




# set_number_caract_dict = {1: "5_0.0001_500_sig", 2: "5_0.001_500", 3: "5_0.0001_500", 4:}
# steps_list = [0, 6000, 12000, 18000, 24000, 30000, 36000, 42000, 50000]
steps_list = [k * 2000 for k in range(26)]
path = f"/scratch/mrmn/gandonb/Exp_StyleGAN/Set_{set_number}/stylegan2_stylegan_dom_256_lat-dim_512_bs_32_0.0002_0.0002_ch-mul_2_vars_rr_noise_True/Instance_{instance_number_list[0]}/log/"
if sets_list:
    path_list = [f"/scratch/mrmn/gandonb/Exp_StyleGAN/Set_{set_num}/stylegan2_stylegan_dom_256_lat-dim_512_bs_32_0.0002_0.0002_ch-mul_2_vars_rr_noise_True/Instance_{instance_number_list[0]}/log/" for set_num in sets_list]
elif len(instance_number_list) > 1:
    path_list = [f"/scratch/mrmn/gandonb/Exp_StyleGAN/Set_{set_number}/stylegan2_stylegan_dom_256_lat-dim_512_bs_32_0.0002_0.0002_ch-mul_2_vars_rr_noise_True/Instance_{instance_num}/log/" for instance_num in instance_number_list]
outpath = f"/scratch/mrmn/gandonb/Presentable/plot_metric/"
steps_str = "_".join([str(step) for step in steps_list])

# filename_distance = f"test_distance_distance_metrics_step_{steps_str}_16384.p"
# filename_standalone = f"test_standalone_standalone_metrics_step_{steps_str}_16384.p"
filename_distance = f"test_distance_{num}_distance_metrics_step_{steps_str}_16384.p"
filename_standalone = f"test_standalone_{num}_standalone_metrics_step_{steps_str}_16384.p"
if num_list:
    filename_distance_list = [f"test_distance_{num}_distance_metrics_step_{steps_str}_16384.p" for num in num_list]
    filename_standalone_list = [f"test_standalone_{num}_standalone_metrics_step_{steps_str}_16384.p" for num in num_list]


def one_set_old(set_number, instance_number, path, outpath, filename_distance, filename_standalone, args):
    with open(f"{path}{filename_standalone}", 'rb') as f:
        standalone_dict = pk.load(f)
    v_dict = {0.01: (-0.1, 0.5), 0.1: (-0.1, 0.5), 0.9: (-0.1, 5), 0.99: (0, 10)}
    for quantile_idx, quantile in enumerate([0.01, 0.1, 0.9, 0.99]):
        outpath_quantile_map = f"{outpath}Set_{set_number}/{num}/quant_map/quantile_{quantile}/"
        outpath_quantile_map_fix = f"{outpath}Set_{set_number}/{num}/quant_map/quantile_{quantile}/fix/"
        make_save_dir(outpath_quantile_map)
        make_save_dir(outpath_quantile_map_fix)
        for idx_step, step in enumerate(steps_list):
            if len(steps_list) == 1:
                quantile_map = standalone_dict["quant_map"][quantile_idx]
            else:
                quantile_map = standalone_dict["quant_map"][idx_step][quantile_idx]
            plt.clf()
            fig, axes = plt.subplots()
            # img = axes.imshow(quantile_map, origin="lower", vmin=v_dict[quantile][0], vmax=v_dict[quantile][1])
            img = axes.imshow(quantile_map, origin="lower")
            fig.colorbar(img)
            fig.savefig(f"{outpath_quantile_map}step_{step}.png")
            plt.close(fig)
            plt.clf()
            fig, axes = plt.subplots()
            img = axes.imshow(quantile_map, origin="lower", vmin=v_dict[quantile][0], vmax=v_dict[quantile][1])
            fig.colorbar(img)
            fig.savefig(f"{outpath_quantile_map_fix}step_{step}.png")
            plt.close(fig)

    with open(f"{path}{filename_distance}", 'rb') as f:
        distance_dict = pk.load(f)
    outpath_pw_W1 = f"{outpath}Set_{set_number}/{num}/pw_W1/"
    make_save_dir(outpath_pw_W1)
    for idx_step, step in enumerate(steps_list):
        if len(steps_list) == 1:
            pw_W1_map = distance_dict["pw_W1"][0][0]
        else:
            pw_W1_map = distance_dict[step]["pw_W1"][0][0]
        plt.clf()
        fig, axes = plt.subplots()
        img = axes.imshow(pw_W1_map, origin="lower")
        fig.colorbar(img)
        fig.savefig(f"{outpath_pw_W1}step_{step}.png")
        plt.close(fig)

    outpath_W1_Center_NUMPY = f"{outpath}Set_{set_number}/{num}/W1_Center_NUMPY/"
    make_save_dir(outpath_W1_Center_NUMPY)
    if len(steps_list) == 1:
        W1_Center_NUMPY_list = [distance_dict["W1_Center_NUMPY"][0] for step in steps_list]
    else:
        W1_Center_NUMPY_list = [distance_dict[step]["W1_Center_NUMPY"][0] for step in steps_list[1:]]
    plt.clf()
    if len(steps_list) == 1:
        plt.plot(steps_list, W1_Center_NUMPY_list, marker=".", label=f"Set {set_number}")
    else:
        plt.plot(steps_list[1:], W1_Center_NUMPY_list, marker=".", label=f"Set {set_number}")
    plt.savefig(f"{outpath_W1_Center_NUMPY}plot.png")

    outpath_W1_random_NUMPY = f"{outpath}Set_{set_number}/{num}/W1_random_NUMPY/"
    make_save_dir(outpath_W1_random_NUMPY)
    if len(steps_list) == 1:
        W1_random_NUMPY_list = [distance_dict["W1_random_NUMPY"][0] for step in steps_list]
    else:
        W1_random_NUMPY_list = [distance_dict[step]["W1_random_NUMPY"][0] for step in steps_list[1:]]
    plt.clf()
    if len(steps_list) == 1:
        plt.plot(steps_list, W1_random_NUMPY_list, marker=".", label=f"Set {set_number}")
    else:
        plt.plot(steps_list[1:], W1_random_NUMPY_list, marker=".", label=f"Set {set_number}")
    plt.savefig(f"{outpath_W1_random_NUMPY}plot.png")
    outpath_SWD = f"{outpath}Set_{set_number}/{num}/SWD/"
    outpath_SWD_fix = f"{outpath}Set_{set_number}/{num}/SWD/fix/"
    make_save_dir(outpath_SWD)
    make_save_dir(outpath_SWD_fix)
    SWD_list = [[] for _ in range(5)] # 4 SWD values + mean SWD value
    for idx_step, step in enumerate(steps_list):
        if len(steps_list) == 1:
            SWD_tensor = distance_dict["SWD_metric_torch"][0]
        else:
            SWD_tensor = distance_dict[step]["SWD_metric_torch"][0]
        for idx_elem, unique_elem_tensor in enumerate(SWD_tensor):
            SWD_list[idx_elem].append(unique_elem_tensor.item())
    for idx_elem, elem_list in enumerate(SWD_list):
        plt.clf()
        plt.plot(steps_list, elem_list, marker=".")
        plt.savefig(f"{outpath_SWD}plot{idx_elem}.png")
        plt.clf()
        plt.plot(steps_list, elem_list, marker=".")
        plt.ylim([0, 200])
        plt.savefig(f"{outpath_SWD_fix}plot{idx_elem}.png")

def compare_within_set(num_list, path_list, instance_number, path, outpath, filename_distance_list, args):
    plt.clf()
    outpath_W1_Center_NUMPY = f"{outpath}Set_{set_number}/compare/W1_Center_NUMPY/"
    make_save_dir(outpath_W1_Center_NUMPY)
    for num, filename_distance in zip(num_list, filename_distance_list):
        with open(f"{path}{filename_distance}", 'rb') as f:
            distance_dict = pk.load(f)
        if len(steps_list) == 1:
            W1_Center_NUMPY_list = [distance_dict["W1_Center_NUMPY"][0] for step in steps_list]
        else:
            W1_Center_NUMPY_list = [distance_dict[step]["W1_Center_NUMPY"][0] for step in steps_list[1:]]
        if len(steps_list) == 1:
            plt.plot(steps_list, W1_Center_NUMPY_list, marker=".", label=f"Num {num}")
        else:
            plt.plot(steps_list[1:], W1_Center_NUMPY_list, marker=".", label=f"Num {num}")
        plt.legend()
        plt.savefig(f"{outpath_W1_Center_NUMPY}plot.png")
    plt.clf()
    outpath_W1_Center_NUMPY = f"{outpath}Set_{set_number}/compare/W1_Center_NUMPY/fix/"
    make_save_dir(outpath_W1_Center_NUMPY)
    for num, filename_distance in zip(num_list, filename_distance_list):
        with open(f"{path}{filename_distance}", 'rb') as f:
            distance_dict = pk.load(f)
        if len(steps_list) == 1:
            W1_Center_NUMPY_list = [distance_dict["W1_Center_NUMPY"][0] for step in steps_list]
        else:
            W1_Center_NUMPY_list = [distance_dict[step]["W1_Center_NUMPY"][0] for step in steps_list[1:]]
        if len(steps_list) == 1:
            plt.plot(steps_list, W1_Center_NUMPY_list, marker=".", label=f"Num {num}")
        else:
            plt.plot(steps_list[1:], W1_Center_NUMPY_list, marker=".", label=f"Num {num}")
        plt.ylim(0, 1500)
        plt.legend()
        plt.savefig(f"{outpath_W1_Center_NUMPY}plot.png")

    plt.clf()
    outpath_W1_random_NUMPY = f"{outpath}Set_{set_number}/compare/W1_random_NUMPY/"
    make_save_dir(outpath_W1_random_NUMPY)
    for num, filename_distance in zip(num_list, filename_distance_list):
        with open(f"{path}{filename_distance}", 'rb') as f:
            distance_dict = pk.load(f)
        if len(steps_list) == 1:
            W1_random_NUMPY_list = [distance_dict["W1_random_NUMPY"][0] for step in steps_list]
        else:
            W1_random_NUMPY_list = [distance_dict[step]["W1_random_NUMPY"][0] for step in steps_list[1:]]
        if len(steps_list) == 1:
            plt.plot(steps_list, W1_random_NUMPY_list, marker=".", label=f"Num {num}")
        else:
            plt.plot(steps_list[1:], W1_random_NUMPY_list, marker=".", label=f"Num {num}")
        plt.legend()
        plt.savefig(f"{outpath_W1_random_NUMPY}plot.png")
    plt.clf()
    outpath_W1_random_NUMPY = f"{outpath}Set_{set_number}/compare/W1_random_NUMPY/fix/"
    make_save_dir(outpath_W1_random_NUMPY)
    for num, filename_distance in zip(num_list, filename_distance_list):
        with open(f"{path}{filename_distance}", 'rb') as f:
            distance_dict = pk.load(f)
        if len(steps_list) == 1:
            W1_random_NUMPY_list = [distance_dict["W1_random_NUMPY"][0] for step in steps_list]
        else:
            W1_random_NUMPY_list = [distance_dict[step]["W1_random_NUMPY"][0] for step in steps_list[1:]]
        if len(steps_list) == 1:
            plt.plot(steps_list, W1_random_NUMPY_list, marker=".", label=f"Num {num}")
        else:
            plt.plot(steps_list[1:], W1_random_NUMPY_list, marker=".", label=f"Num {num}")
        plt.ylim(0, 1500)
        plt.legend()
        plt.savefig(f"{outpath_W1_random_NUMPY}plot.png")
        
    outpath_SWD = f"{outpath}Set_{set_number}/compare/SWD/"
    outpath_SWD_fix = f"{outpath}Set_{set_number}/compare/SWD/fix/"
    make_save_dir(outpath_SWD)
    make_save_dir(outpath_SWD_fix)
    SWD_list = [[[] for _ in range(5)] for _ in steps_list] # range(5) because 4 SWD values + mean SWD value
    for SWD_idx in range(5):
        plt.clf()
        for num, filename_distance in zip(num_list, filename_distance_list):
            with open(f"{path}{filename_distance}", 'rb') as f:
                distance_dict = pk.load(f)
            if len(steps_list) == 1:
                SWD_list = [distance_dict["SWD_metric_torch"][0][SWD_idx].item() for step in steps_list]
            else:
                SWD_list = [distance_dict[step]["SWD_metric_torch"][0][SWD_idx].item() for step in steps_list]
            
            plt.plot(steps_list, SWD_list, marker=".", label=f"Num {num}")
        plt.legend()
        plt.savefig(f"{outpath_SWD}plot_{SWD_idx + 1}.png")
    for SWD_idx in range(5):
        plt.clf()
        for num, filename_distance in zip(num_list, filename_distance_list):
            with open(f"{path}{filename_distance}", 'rb') as f:
                distance_dict = pk.load(f)
            if len(steps_list) == 1:
                SWD_list = [distance_dict["SWD_metric_torch"][0][SWD_idx].item() for step in steps_list]
            else:
                SWD_list = [distance_dict[step]["SWD_metric_torch"][0][SWD_idx].item() for step in steps_list]
            plt.plot(steps_list, SWD_list, marker=".", label=f"Num {num}")
        plt.legend()
        plt.ylim([0, 400])
        plt.savefig(f"{outpath_SWD_fix}plot_{SWD_idx + 1}.png")

def compare_instance(set_number, path_list, instance_number_list, outpath, filename_distance, filename_standalone, args):
    plt.clf()
    outpath_W1_Center_NUMPY = f"{outpath}Set_{set_number}/compare_instances_{'_'.join([str(instance_num) for instance_num in instance_number_list])}/W1_Center_NUMPY/"
    make_save_dir(outpath_W1_Center_NUMPY)
    for instance_number, path in zip(instance_number_list, path_list):
        with open(f"{path}{filename_distance}", 'rb') as f:
            distance_dict = pk.load(f)
        W1_Center_NUMPY_list = [distance_dict[step]["W1_Center_NUMPY"][0] for step in steps_list[1:]]
        plt.plot(steps_list[1:], W1_Center_NUMPY_list, marker=".", label=f"Instance {instance_number}")
    plt.legend()
    plt.savefig(f"{outpath_W1_Center_NUMPY}plot.png")

    plt.clf()
    outpath_W1_random_NUMPY = f"{outpath}Set_{set_number}/compare_instances_{'_'.join([str(instance_num) for instance_num in instance_number_list])}/W1_random_NUMPY/"
    make_save_dir(outpath_W1_random_NUMPY)
    for instance_number, path in zip(instance_number_list, path_list):
        with open(f"{path}{filename_distance}", 'rb') as f:
            distance_dict = pk.load(f)
        W1_random_NUMPY_list = [distance_dict[step]["W1_random_NUMPY"][0] for step in steps_list[1:]]
        plt.plot(steps_list[1:], W1_random_NUMPY_list, marker=".", label=f"Instance {instance_number}")
    plt.legend()
    plt.savefig(f"{outpath_W1_random_NUMPY}plot.png")
    
    outpath_SWD = f"{outpath}Set_{set_number}/compare_instances_{'_'.join([str(instance_num) for instance_num in instance_number_list])}/SWD/"
    outpath_SWD_fix = f"{outpath}Set_{set_number}/compare_instances_{'_'.join([str(instance_num) for instance_num in instance_number_list])}/SWD/fix/"
    make_save_dir(outpath_SWD)
    make_save_dir(outpath_SWD_fix)
    SWD_list = [[[] for _ in range(5)] for _ in steps_list] # range(5) because 4 SWD values + mean SWD value
    for SWD_idx in range(5):
        plt.clf()
        for instance_number, path in zip(instance_number_list, path_list):
            with open(f"{path}{filename_distance}", 'rb') as f:
                distance_dict = pk.load(f)
            SWD_list = [distance_dict[step]["SWD_metric_torch"][0][SWD_idx].item() for step in steps_list]
            plt.plot(steps_list, SWD_list, marker=".", label=f"Instance {instance_number}")
        plt.legend()
        plt.savefig(f"{outpath_SWD}plot_{SWD_idx + 1}.png")
    for SWD_idx in range(5):
        plt.clf()
        for instance_number, path in zip(instance_number_list, path_list):
            with open(f"{path}{filename_distance}", 'rb') as f:
                distance_dict = pk.load(f)
            SWD_list = [distance_dict[step]["SWD_metric_torch"][0][SWD_idx].item() for step in steps_list]
            plt.plot(steps_list, SWD_list, marker=".", label=f"Instance {instance_number}")
        plt.legend()
        plt.ylim([0, 400])
        plt.savefig(f"{outpath_SWD_fix}plot_{SWD_idx + 1}.png")


def compare_sets(sets_list, path_list, instance_number, outpath, filename_distance, filename_standalone, args):
    plt.clf()
    outpath_W1_Center_NUMPY = f"{outpath}compare_sets_{'_'.join([str(set_num) for set_num in sets_list])}/W1_Center_NUMPY/"
    make_save_dir(outpath_W1_Center_NUMPY)
    for set_number, path in zip(sets_list, path_list):
        with open(f"{path}{filename_distance}", 'rb') as f:
            distance_dict = pk.load(f)
        W1_Center_NUMPY_list = [distance_dict[step]["W1_Center_NUMPY"][0] for step in steps_list[1:]]
        plt.plot(steps_list[1:], W1_Center_NUMPY_list, marker=".", label=f"Set {set_number}")
    plt.legend()
    plt.savefig(f"{outpath_W1_Center_NUMPY}plot.png")

    plt.clf()
    outpath_W1_random_NUMPY = f"{outpath}compare_sets_{'_'.join([str(set_num) for set_num in sets_list])}/W1_random_NUMPY/"
    make_save_dir(outpath_W1_random_NUMPY)
    for set_number, path in zip(sets_list, path_list):
        with open(f"{path}{filename_distance}", 'rb') as f:
            distance_dict = pk.load(f)
        W1_random_NUMPY_list = [distance_dict[step]["W1_random_NUMPY"][0] for step in steps_list[1:]]
        plt.plot(steps_list[1:], W1_random_NUMPY_list, marker=".", label=f"Set {set_number}")
    plt.legend()
    plt.savefig(f"{outpath_W1_random_NUMPY}plot.png")
    
    outpath_SWD = f"{outpath}compare_sets_{'_'.join([str(set_num) for set_num in sets_list])}/SWD/"
    outpath_SWD_fix = f"{outpath}compare_sets_{'_'.join([str(set_num) for set_num in sets_list])}/SWD/fix/"
    make_save_dir(outpath_SWD)
    make_save_dir(outpath_SWD_fix)
    SWD_list = [[[] for _ in range(5)] for _ in steps_list] # range(5) because 4 SWD values + mean SWD value
    for SWD_idx in range(5):
        plt.clf()
        for set_number, path in zip(sets_list, path_list):
            with open(f"{path}{filename_distance}", 'rb') as f:
                distance_dict = pk.load(f)
            SWD_list = [distance_dict[step]["SWD_metric_torch"][0][SWD_idx].item() for step in steps_list]
            plt.plot(steps_list, SWD_list, marker=".", label=f"Set {set_number}")
        plt.legend()
        plt.savefig(f"{outpath_SWD}plot_{SWD_idx + 1}.png")
    for SWD_idx in range(5):
        plt.clf()
        for set_number, path in zip(sets_list, path_list):
            with open(f"{path}{filename_distance}", 'rb') as f:
                distance_dict = pk.load(f)
            SWD_list = [distance_dict[step]["SWD_metric_torch"][0][SWD_idx].item() for step in steps_list]
            plt.plot(steps_list, SWD_list, marker=".", label=f"Set {set_number}")
        plt.legend()
        plt.ylim([0, 400])
        plt.savefig(f"{outpath_SWD_fix}plot_{SWD_idx + 1}.png")

    # for set_number, path in zip(sets_list, path_list):
    #     with open(f"{path}{filename_distance}", 'rb') as f:
    #             distance_dict = pk.load(f)
    #     for idx_step, step in enumerate(steps_list):
    #         SWD_tensor = distance_dict[step]["SWD_metric_torch"][0]
    #         for idx_elem, unique_elem_tensor in enumerate(SWD_tensor):
    #             SWD_list[set_number][idx_elem].append(unique_elem_tensor.item())
    # for set_idx, set_number in sets_list:
    #     for idx_elem, elem_list in enumerate(SWD_list[set_idx]):
    #         plt.plot(steps_list, elem_list)
    #         plt.savefig(f"{outpath_SWD}plot{idx_elem}.png")
    # plt.clf()
    # for set_idx, set_number in sets_list:
    #     for idx_elem, elem_list in enumerate(SWD_list[set_idx]):
    #         plt.plot(steps_list, elem_list)
    #         plt.ylim([0, 200])
    #         plt.savefig(f"{outpath_SWD_fix}plot{idx_elem}.png")
    # plt.clf()


# if sets_list:
#     compare_sets(sets_list, path_list, instance_number[0], outpath, filename_distance, filename_standalone, args)
# elif len(instance_number_list) > 1:
#     compare_instance(set_number, path_list, instance_number_list, outpath, filename_distance, filename_standalone, args)
# else:
#     if num_list:
#         compare_within_set(num_list, path, instance_number[0], path, outpath, filename_distance_list, args)
#     else:
#         one_set(set_number, instance_number[0], path, outpath, filename_distance, filename_standalone, args)