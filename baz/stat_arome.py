import os
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import torch

path = os.path.join("/scratch", "work", "gandonb", "Exp_StyleGAN", "AROME", "pickles")
sample_nums = [128 * 2 ** sample_idx for sample_idx in range(7)]
prog_num = 40

W1_Center_array = np.empty((len(sample_nums), prog_num))
W1_random_array = np.empty((len(sample_nums), prog_num))
for sample_num_idx, sample_num in enumerate(sample_nums):
    stat_file = f"AROME_W1_Center_W1_random_rr_u_v_t2m_dom_256{f'_{sample_num}' * prog_num}.p"
    with open(os.path.join(path, stat_file), 'rb') as pkfile:
        result_dict = pk.load(pkfile)
    W1_Center_array[sample_num_idx] = np.array([result_dict[prog_idx]["W1_Center_NUMPY"][0] for prog_idx in range(prog_num)])
    W1_random_array[sample_num_idx] = np.array([result_dict[prog_idx]["W1_random_NUMPY"][0] for prog_idx in range(prog_num)])

W1_Center_mean_array = W1_Center_array.mean(axis=1)
plt.clf()
plt.plot(sample_nums, W1_Center_mean_array, marker='.', label="W1 center crop means")
plt.title("W1 center crop mean over 40 computations")
plt.ylabel("W1 center crop value")
plt.xlabel("Number of samples it is computed on")
plt.legend()
plt.grid()
save_dir = os.path.join(path, "distances", "W1_center")
os.makedirs(save_dir, exist_ok=True)
filename = "W1_center_mean_evolution_w_sample_num.png"
plt.savefig(os.path.join(save_dir, filename))
np.save(os.path.join(save_dir, filename), W1_Center_mean_array)

W1_random_mean_array = W1_random_array.mean(axis=1)
plt.clf()
plt.plot(sample_nums, W1_random_mean_array, marker='.', label="W1 random crop means")
plt.title("W1 random crop mean over 40 computations")
plt.ylabel("W1 random crop value")
plt.xlabel("Number of samples it is computed on")
plt.legend()
plt.grid()
save_dir = os.path.join(path, "distances", "W1_random")
os.makedirs(save_dir, exist_ok=True)
filename = "W1_random_mean_evolution_w_sample_num.png"
plt.savefig(os.path.join(save_dir, filename))
np.save(os.path.join(save_dir, filename), W1_random_mean_array)

W1_Center_std_array = W1_Center_array.std(axis=1)
plt.clf()
plt.plot(sample_nums, W1_Center_std_array, marker='.', label="W1 center crop standard deviations")
plt.title("W1 center crop standard deviations over 40 computations")
plt.ylabel("W1 center crop value")
plt.xlabel("Number of samples it is computed on")
plt.legend()
plt.grid()
save_dir = os.path.join(path, "distances", "W1_center")
os.makedirs(save_dir, exist_ok=True)
filename = "W1_center_std_evolution_w_sample_num.png"
plt.savefig(os.path.join(save_dir, filename))
np.save(os.path.join(save_dir, filename), W1_Center_std_array)

W1_random_std_array = W1_random_array.std(axis=1)
plt.clf()
plt.plot(sample_nums, W1_random_std_array, marker='.', label="W1 random crop standard deviations")
plt.title("W1 random crop standard deviations over 40 computations")
plt.ylabel("W1 random crop value")
plt.xlabel("Number of samples it is computed on")
plt.legend()
plt.grid()
save_dir = os.path.join(path, "distances", "W1_random")
os.makedirs(save_dir, exist_ok=True)
filename = "W1_random_std_evolution_w_sample_num.png"
plt.savefig(os.path.join(save_dir, filename))
np.save(os.path.join(save_dir, filename), W1_random_std_array)