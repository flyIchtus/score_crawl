import pickle as pk

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

path = f"/scratch/mrmn/gandonb/Exp_StyleGAN/Set_9/stylegan2_stylegan_dom_256_lat-dim_512_bs_32_0.0002_0.0002_ch-mul_2_vars_rr_noise_True/Instance_2/log/"

filename = "AROME_test_distance_1_distance_metrics_step_0_2000_4000_6000_8000_10000_12000_14000_16000_18000_20000_22000_24000_26000_28000_30000_32000_34000_36000_38000_40000_42000_44000_46000_48000_50000_16384.p"

steps_list = [k * 2000 for k in range(26)]

outpath = f"/scratch/mrmn/gandonb/Presentable/plot_metric/"
outpath_W1_Center_NUMPY = f"{outpath}Set_AROME/W1_Center_NUMPY/"

with open(f"{path}{filename}", 'rb') as pkfile:
    arome_dict = pk.load(pkfile)

W1_Center_NUMPY_list = []
for step in steps_list:
    W1_Center_NUMPY_list.append(arome_dict[step]["W1_Center_NUMPY"][0])
plt.clf()
plt.plot(steps_list, W1_Center_NUMPY_list, marker=".")
plt.savefig(f"{outpath_W1_Center_NUMPY}plot.png")