import matplotlib.pyplot as plt
import numpy as np
import os

img = False

main_path = "/scratch/work/gandonb/Presentable/plot_metric/"
var_names = ["rr", "u", "v", "t2m"]
dict_caract_set = {
    11: "bs 8, tanhF",
    32: "bs 128, tanhT",
    33: "bs 8, tanhT",
    34: "bs 32, tanhT",
    100: "bs 32, tanhT, d_optim_modif",
    101: "bs 32, tanhT, d_optim_modif",
    102: "bs 32, tanhF, d_optim_modif",
    103: "bs 32, tanhF, d_optim_modif",
    105: "bs 16, tanF, d_optim_modif, quant",
    555: "bs 1, tanhF",
    556: "bs 1, tanhT",
    557: "555 like but optim.step()",
    558: "555 like",
    1001: "bs 32, tanF, d_optim_modif, quant",
    1003: "GAN",
    1005: "GAN IS",
}
dict_steps_set = {
    11: 190000,
    32: 120000,
    33: 110000,
    34: 230000,
    100: 100000,
    101: 100000,
    102: 100000,
    103: 100000,
    105: 100000,
    555: 500000,
    556: 100000,
    557: 370000,
    558: 370000,
    1001: 240000,
    1003: 320000,
    1005: 280000,
}
dict_batch_size = {
    11: 8,
    32: 128,
    33: 8,
    34: 32,
    100: 32,
    101: 32,
    102: 32,
    103: 32,
    105: 16,
    555: 1,
    556: 1,
    557: 1,
    558: 1,
    1001: 32,
    1003: 32,
    1005: 32,
}
dict_step = {
    11: 10000,
    32: 10000,
    33: 10000,
    34: 10000,
    100: 10000,
    101: 10000,
    102: 10000,
    103: 10000,
    555: 10000,
    556: 10000,
    557: 10000,
    558: 50000,
    1001: 20000,
    1003: 20000,
    1005: 20000,
}
dict_color_set = {
    11: "orange",
    32: "green",
    33: "red",
    34: "green",
    100: "orange",
    101: "purple",
    102: "blue",
    103: "cyan",
    105: "red",
    555: "blue",
    556: "yellow",
    557: "red",
    558: "cyan",
    1001: "green",
    1003: "green",
    1005: "red",
}
set_list = [1003, 1005]

step_min = 0
step_max = max(dict_steps_set.values())
step_max = 320000
step_space = 20000

def compare_W1(set_list):
    plt.clf()
    for set_num in set_list:
        W1_random_plot = np.load(os.path.join(main_path, f"Set_{set_num}", "Instance_1", "W1_random", "W1_random.npy"))
        plt.plot(W1_random_plot[0][1:] * (dict_batch_size[set_num] if img else 1), W1_random_plot[1][1:], marker=".", mec=dict_color_set[set_num], color=dict_color_set[set_num], label=f"{set_num}: {dict_caract_set[set_num]}")
    W1_random_plot = np.load(os.path.join(main_path, f"AROME", "Instance_1", "W1_random", "W1_random.npy"))
    plt.plot([step * 10000 for step in range((step_max - step_min) // step_space)], [W1_random_plot[1] for _ in range((step_max - step_min) // step_space)], marker="o", mfc="purple", mec="purple", color="purple", label="AROME baseline")
    plt.title("W1 random crop")
    if img:
        plt.xlabel("AROME samples seen")
    else:
        plt.xlabel("Steps")
    plt.ylabel("W1 value")
    if img:
        plt.xscale("log")
    plt.legend()
    save_dir = os.path.join("compare", "W1_random")
    os.makedirs(os.path.join(main_path, save_dir), exist_ok=True)
    plt.savefig(os.path.join(main_path, save_dir, "W1_random.png"))

    plt.clf()
    for set_num in set_list:
        W1_Center_plot = np.load(os.path.join(main_path, f"Set_{set_num}", "Instance_1", "W1_Center", "W1_Center.npy"))
        plt.plot(W1_Center_plot[0][1:] * (dict_batch_size[set_num] if img else 1), W1_Center_plot[1][1:], marker=".", mec=dict_color_set[set_num], color=dict_color_set[set_num], label=f"{set_num}: {dict_caract_set[set_num]}")
    W1_Center_plot = np.load(os.path.join(main_path, f"AROME", "Instance_1", "W1_Center", "W1_Center.npy"))
    plt.plot([step * 10000 for step in range((step_max - step_min) // step_space)], [W1_Center_plot[1] for _ in range((step_max - step_min) // step_space)], marker="o", mfc="purple", mec="purple", color="purple", label="AROME baseline")
    plt.title("W1 Center crop")
    if img:
        plt.xlabel("AROME samples seen")
    else:
        plt.xlabel("Steps")
    plt.ylabel("W1 value")
    if img:
        plt.xscale("log")
    plt.legend()
    save_dir = os.path.join("compare", "W1_Center")
    os.makedirs(os.path.join(main_path, save_dir), exist_ok=True)
    plt.savefig(os.path.join(main_path, save_dir, "W1_Center.png"))

def compare_SWD(set_list):
    SWD_scale_list = [128, 64, 32, 16, "mean"]
    for SWD_scale in SWD_scale_list:
        plt.clf()
        for set_num in set_list:
            SWD_plot = np.load(os.path.join(main_path, f"Set_{set_num}", "Instance_1", "SWD_torch", f"SWD_{SWD_scale}.npy"))
            plt.plot(SWD_plot[0][1:] * (dict_batch_size[set_num] if img else 1), SWD_plot[1][1:], marker=".", mec=dict_color_set[set_num], color=dict_color_set[set_num], label=f"{set_num}: {dict_caract_set[set_num]}")
        SWD_plot = np.load(os.path.join(main_path, f"AROME", "Instance_1", "SWD_torch", f"SWD_{SWD_scale}.npy"))
        plt.plot([step * 10000 for step in range((step_max - step_min) // step_space)], [SWD_plot[1] for _ in range((step_max - step_min) // step_space)], marker="o", mfc="purple", mec="purple", color="purple", label="AROME baseline")
        plt.title(f"SWD {SWD_scale}")
        if img:
            plt.xlabel("AROME samples seen")
        else:
            plt.xlabel("Steps")
        plt.ylabel("SWD value")
        if img:
            plt.xscale("log")
        plt.legend()
        save_dir = os.path.join("compare", "SWD")
        os.makedirs(os.path.join(main_path, save_dir), exist_ok=True)
        plt.savefig(os.path.join(main_path, save_dir, f"SWD_{SWD_scale}.png"))

def compare_spectra(set_list):
    dict_ylim = {
        "rr": (10e-8, 10e-1),
        "u": (10e-7, 10e-1),
        "v": (10e-7, 10e-1),
        "t2m": (10e-7, 10e-1),
    }
    for var_name in var_names:
        for step in range(step_min, step_max + 1, step_space):
            plt.clf()
            spectral_plot = np.load(os.path.join(main_path, f"AROME", "Instance_1", "spectral_compute", var_name, f"spectral_compute_{var_name}_step_0.npy"))
            plt.plot(spectral_plot[0], spectral_plot[1], marker="o", mfc="purple", mec="purple", color="purple", label="AROME baseline")
            for set_num in set_list:
                if set_num == 558:
                    step = int(step / 10000 * 50000)
                if step <= dict_steps_set[set_num]:
                    spectral_plot = np.load(os.path.join(main_path, f"Set_{set_num}", "Instance_1", "spectral_compute", var_name, f"spectral_compute_{var_name}_step_{step}.npy"))
                    plt.plot(spectral_plot[0], spectral_plot[1], marker=".", mec=dict_color_set[set_num], color=dict_color_set[set_num], label=f"{set_num}: {dict_caract_set[set_num]}")
            plt.title(f"spectrum at step {step} for {var_name}")
            plt.xlabel("frequency")
            plt.ylabel("Variance of FFT")
            plt.xscale("log")
            plt.yscale("log")
            plt.ylim(dict_ylim[var_name])
            plt.legend()
            save_dir = os.path.join("compare", "spectra", var_name)
            os.makedirs(os.path.join(main_path, save_dir), exist_ok=True)
            plt.savefig(os.path.join(main_path, save_dir, f"spectrum_at_step_{step}.png"))

print("Compare W1...")
compare_W1(set_list)

print("Compare SWD...")
compare_SWD(set_list)

print("Compare spectra...")
compare_spectra(set_list)
