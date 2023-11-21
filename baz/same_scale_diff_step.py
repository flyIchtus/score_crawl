import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

dict_caract_set = {
    11: "bs 8, tanhF",
    32: "bs 128, tanhT",
    33: "bs 8, tanhT",
    34: "bs 32, tanhT",
    100: "bs 32, tanhT, d_optim_modif",
    101: "bs 32, tanhT, d_optim_modif",
    102: "bs 32, tanhF, d_optim_modif",
    103: "bs 32, tanhF, d_optim_modif",
    555: "bs 1, tanhF",
    556: "bs 1, tanhT",
    557: "555 like but optim.step()",
    558: "555 like",
    1001: "GAN",
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
    555: 500000,
    556: 100000,
    557: 370000,
    558: 370000,
    1001: 240000,
    1003: 270000,
    1005: 220000,
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
    1001: 10000,
    1003: 10000,
    1005: 10000,
}
dict_color_set = {
    11: "orange",
    32: "green",
    33: "red",
    34: "pink",
    100: "orange",
    101: "red",
    102: "blue",
    103: "cyan",
    555: "blue",
    556: "yellow",
    557: "red",
    558: "cyan",
    1001: "red",
    1003: "green",
    1005: "red",
}
set_list = [1003, 1005]

arome_is = True
comp = True

set_num = 146

step_max = 270000

path = "/scratch/work/gandonb/making_stats/"
save_dir = f"{path}AROME_Set_{set_num}/"
# path = "/scratch/work/gandonb/making_stats/"
# save_dir = f"{path}~Set_{set_num}/pictures/"
os.makedirs(save_dir, exist_ok=True)

stat_name = "area_proportion"
extension = ".json"
save_dir_area_prop = f"{save_dir}area_proportion/"
os.makedirs(save_dir_area_prop, exist_ok=True)
for step in range(0, step_max + 1, 10000):
    print(step)
    if comp:
        stat_name = "area_proportion"
        extension = ".json"
        plt.clf()
        with open(f"{path}~AROMEvs/AROME{'_IS' if arome_is else ''}/{stat_name}{extension}") as logfile_arome:
            dictio_arome = json.load(logfile_arome)
            dictio_arome = {float(key): value for key, value in dictio_arome.items()}
        keys = np.array(sorted(list(map(float, dictio_arome.keys()))))
        values = np.array([dictio_arome[key] for key in keys])
        plt.plot(keys, values, label=f"AROME{'_IS' if arome_is else ''}", color="purple")
        with open(f"{path}~AROMEvs/AROME/{stat_name}{extension}") as logfile_arome:
            dictio_arome = json.load(logfile_arome)
            dictio_arome = {float(key): value for key, value in dictio_arome.items()}
        keys = np.array(sorted(list(map(float, dictio_arome.keys()))))
        values = np.array([dictio_arome[key] for key in keys])
        plt.plot(keys, values, label="AROME", color="black")
        for set_num in set_list:
            try:
                with open(f"{path}~Set_{set_num}/step_{step}/~stats/{stat_name}{extension}") as logfile_gan:
                    dictio_gan = json.load(logfile_gan)
                    dictio_gan = {float(key): value for key, value in dictio_gan.items()}
            except FileNotFoundError as err:
                print(f'Set {set_num} has no data for step {step}, SKIPPING')
            keys = np.array(sorted(list(map(float, dictio_gan.keys()))))
            values = np.array([dictio_gan[key] for key in keys])
            plt.plot(keys, values, label=dict_caract_set[set_num], color=dict_color_set[set_num])
        plt.ylim(10e-9, 10e-3)
        plt.yscale("log")
        plt.xlabel("s_rr")
        plt.ylabel("Area proportion (log10)")
        plt.title(f"Area proportion for precipitation >= s_rr GANs vs AROME{' IS' if arome_is else ''} step {step}")
        plt.legend()
        plt.savefig(f"{save_dir_area_prop}comp_area_proportion{'_IS' if arome_is else ''}_step_{step}.png")
    else:
        stat_name = "area_proportion"
        extension = ".json"
        with open(f"{path}~AROMEvs/AROME{'_IS' if arome_is else ''}/{stat_name}{extension}") as logfile_arome:
            dictio_arome = json.load(logfile_arome)

        with open(f"{path}~Set_{set_num}/step_{dict_steps_set[set_num]}/~stats/{stat_name}{extension}") as logfile_gan:
            dictio_gan = json.load(logfile_gan)
        plt.clf()
        for dictio, name, color in zip([dictio_arome, dictio_gan], [f"AROME{'_IS' if arome_is else ''}", "GAN"], ["purple", "b"]):
            keys = np.array(sorted(list(map(float, dictio.keys()))))
            values = np.array([dictio[str(key)] for key in keys])
            plt.plot(keys, values, label=name, color=color)
        plt.yscale("log")
        plt.xlabel("s_rr")
        plt.ylabel("Area proportion (log10)")
        plt.title(f"Area proportion for precipitation >= s_rr GANs vs AROME{' IS' if arome_is else ''}")
        plt.legend()
        plt.savefig(f"{save_dir}comp_area_proportion{'_IS' if arome_is else ''}.png")
"""
stat_name = "extracted_values"
extension = ".json"

v_min, v_max = 0, 20

with open(f"{path}~AROMEvs/AROME{'_IS' if arome_is else ''}/{stat_name}{extension}") as logfile_arome:
    dictio_arome = json.load(logfile_arome)
with open(f"{path}~Set_{set_num}/step_{dict_steps_set[set_num]}/~stats/{stat_name}{extension}") as logfile_gan:
    dictio_gan = json.load(logfile_gan)
plt.clf()
keys = np.array(sorted(list(map(float, dictio_arome.keys()))))
values = np.array([dictio_arome[str(key)] for key in keys])
plt.hist(x=keys, weights=values, density=True, color="purple", bins=(v_max-v_min) * 2, range=(v_min, v_max), label=f"AROME{'_IS' if arome_is else ''}", alpha=0.3)
keys = np.array(sorted(list(map(float, dictio_gan.keys()))))
values = np.array([dictio_gan[str(key)] for key in keys])
plt.hist(x=keys, weights=values, density=True, color="b", bins=(v_max-v_min) * 2, range=(v_min, v_max), label=f"GAN, {dict_caract_set[set_num]}", alpha=0.5)
plt.xlabel("s_rr")
plt.ylabel(stat_name)
plt.title(f"GAN vs AROME{' IS' if arome_is else ''}")
plt.legend()
plt.savefig(f"{save_dir}comp_{stat_name}_{v_min}_{v_max}{'_IS' if arome_is else ''}.png")

v_min, v_max = 2, 20

with open(f"{path}~AROMEvs/AROME{'_IS' if arome_is else ''}/{stat_name}{extension}") as logfile_arome:
    dictio_arome = json.load(logfile_arome)
with open(f"{path}~Set_{set_num}/step_{dict_steps_set[set_num]}/~stats/{stat_name}{extension}") as logfile_gan:
    dictio_gan = json.load(logfile_gan)
plt.clf()
keys = np.array(sorted(list(map(float, dictio_arome.keys()))))
values = np.array([dictio_arome[str(key)] for key in keys])
plt.hist(x=keys, weights=values, density=True, color="purple", bins=(v_max-v_min) * 2, range=(v_min, v_max), label=f"AROME{'_IS' if arome_is else ''}", alpha=0.3)
keys = np.array(sorted(list(map(float, dictio_gan.keys()))))
values = np.array([dictio_gan[str(key)] for key in keys])
plt.hist(x=keys, weights=values, density=True, color="b", bins=(v_max-v_min) * 2, range=(v_min, v_max), label="GAN", alpha=0.5)
plt.xlabel("s_rr")
plt.ylabel(stat_name)
plt.legend()
plt.title(f"GAN vs AROME{' IS' if arome_is else ''}")
plt.savefig(f"{save_dir}comp_{stat_name}_{v_min}_{v_max}{'_IS' if arome_is else ''}.png")

v_min, v_max = 20, 70

with open(f"{path}~AROMEvs/AROME{'_IS' if arome_is else ''}/{stat_name}{extension}") as logfile_arome:
    dictio_arome = json.load(logfile_arome)
with open(f"{path}~Set_{set_num}/step_{dict_steps_set[set_num]}/~stats/{stat_name}{extension}") as logfile_gan:
    dictio_gan = json.load(logfile_gan)
plt.clf()
keys = np.array(sorted(list(map(float, dictio_arome.keys()))))
values = np.array([dictio_arome[str(key)] for key in keys])
plt.hist(x=keys, weights=values, density=True, color="purple", bins=(v_max-v_min) * 2, range=(v_min, v_max), label=f"AROME{'_IS' if arome_is else ''}", alpha=0.3)
keys = np.array(sorted(list(map(float, dictio_gan.keys()))))
values = np.array([dictio_gan[str(key)] for key in keys])
plt.hist(x=keys, weights=values, density=True, color="b", bins=(v_max-v_min) * 2, range=(v_min, v_max), label="GAN", alpha=0.5)
plt.xlabel("s_rr")
plt.title(f"GAN vs AROME{' IS' if arome_is else ''}")
plt.ylabel(stat_name)
plt.legend()
plt.savefig(f"{save_dir}comp_{stat_name}_{v_min}_{v_max}{'_IS' if arome_is else ''}.png")
"""
