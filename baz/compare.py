import matplotlib.pyplot as plt
import numpy as np
import os

main_path = "/scratch/work/gandonb/Presentable/plot_metric/"
var_names = ["rr", "u", "v", "t2m"]
dict_caract_set = {
    1: "loglog, minmax, gn, ppx",
    2: "loglog, minmax, gn",
    3: "loglog, minmax",
    4: "loglog",
    5: "minmax",
    6: "loglog, minmax, ppx",
    7: "sym, minmax, gn, ppx",
    10: "779 + bs 8",
    11: "779 + bs 8 + lr 0.002",
    778: "NO IS, loglog, minmax",
    779: "NO IS, log, minmax",
}
dict_steps_set = {
    1: 120000,
    2: 120000,
    3: 100000,
    4: 100000,
    5: 80000,
    6: 80000,
    7: 100000,
    10: 110000,
    11: 110000,
    778: 120000,
    779: 120000,
}
dict_color_set = {
    1: "orange",
    2: "green",
    3: "red",
    4: "black",
    5: "yellow",
    6: "blue",
    7: "pink",
    10: "green",
    11: "red",
    778: "yellow",
    779: "pink",
}
step_min = 0
step_max = max(dict_steps_set.values())
step_space = 10000

set_list = [778, 779, 10, 11]
def compare_W1(set_list):
    plt.clf()
    for set_num in set_list:
        W1_random_plot = np.load(os.path.join(main_path, f"Set_{set_num}", "Instance_1", "W1_random", "W1_random.npy"))
        plt.plot(W1_random_plot[0][1:], W1_random_plot[1][1:], marker=".", mec=dict_color_set[set_num], color=dict_color_set[set_num], label=f"{set_num}: {dict_caract_set[set_num]}")
    W1_random_plot = np.load(os.path.join(main_path, f"AROME", "Instance_1", "W1_random", "W1_random.npy"))
    plt.plot([step * 10000 for step in range((step_max - step_min) // step_space)], [W1_random_plot[1] for _ in range((step_max - step_min) // step_space)], marker="o", mfc="purple", mec="purple", color="purple", label="AROME baseline")
    plt.title("W1 random crop")
    plt.xlabel("Steps")
    plt.ylabel("W1 value")
    plt.legend()
    save_dir = os.path.join("compare", "W1_random")
    os.makedirs(os.path.join(main_path, save_dir), exist_ok=True)
    plt.savefig(os.path.join(main_path, save_dir, "W1_random.png"))

    plt.clf()
    for set_num in set_list:
        W1_Center_plot = np.load(os.path.join(main_path, f"Set_{set_num}", "Instance_1", "W1_Center", "W1_Center.npy"))
        plt.plot(W1_Center_plot[0][1:], W1_Center_plot[1][1:], marker=".", mec=dict_color_set[set_num], color=dict_color_set[set_num], label=f"{set_num}: {dict_caract_set[set_num]}")
    W1_Center_plot = np.load(os.path.join(main_path, f"AROME", "Instance_1", "W1_Center", "W1_Center.npy"))
    plt.plot([step * 10000 for step in range((step_max - step_min) // step_space)], [W1_Center_plot[1] for _ in range((step_max - step_min) // step_space)], marker="o", mfc="purple", mec="purple", color="purple", label="AROME baseline")
    plt.title("W1 Center crop")
    plt.xlabel("Steps")
    plt.ylabel("W1 value")
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
            plt.plot(SWD_plot[0], SWD_plot[1], marker=".", mec=dict_color_set[set_num], color=dict_color_set[set_num], label=f"{set_num}: {dict_caract_set[set_num]}")
        SWD_plot = np.load(os.path.join(main_path, f"AROME", "Instance_1", "SWD_torch", f"SWD_{SWD_scale}.npy"))
        plt.plot([step * 10000 for step in range((step_max - step_min) // step_space)], [SWD_plot[1] for _ in range((step_max - step_min) // step_space)], marker="o", mfc="purple", mec="purple", color="purple", label="AROME baseline")
        plt.title(f"SWD {SWD_scale}")
        plt.xlabel("Steps")
        plt.ylabel("SWD value")
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