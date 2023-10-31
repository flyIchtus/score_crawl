import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--set", type=int, default=1, help="Set to compare")
args = parser.parse_args()
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

arome_is = False
comp = True

set_num = args.set
set_list = [11]

path = "/scratch/work/gandonb/making_stats/~AROMEvs/"
save_dir = f"{path}AROME_Set_{set_num}/"
# path = "/scratch/work/gandonb/making_stats/"
# save_dir = f"{path}~Set_{set_num}/pictures/"
os.makedirs(save_dir, exist_ok=True)

stat_name = "area_proportion"
extension = ".json"

"""
with open(f"{path}AROME/{stat_name}{extension}") as logfile_arome:
    dictio_arome = json.load(logfile_arome)
with open(f"{path}GAN/Set_{set_num}/{stat_name}{extension}") as logfile_gan:
    dictio_gan = json.load(logfile_gan)
plt.clf()
keys = np.array(sorted(list(map(int, dictio_arome.keys()))))
values = np.array([dictio_arome[str(key)] for key in keys])
plt.plot(keys, values, label="AROME")
keys = np.array(sorted(list(map(int, dictio_gan.keys()))))
values = np.array([dictio_gan[str(key)] for key in keys])
plt.plot(keys, values, label="GAN")
plt.yscale("log")
plt.xlabel("s_rr")
plt.ylabel(stat_name)
plt.legend()
plt.savefig(f"{path}MIX_{stat_name}.png")


stat_name = "extracted_values"
extension = ".json"

v_min, v_max = 40, 90

with open(f"{path}AROME/{stat_name}{extension}") as logfile_arome:
    dictio_arome = json.load(logfile_arome)
with open(f"{path}GAN/Set_{set_num}/{stat_name}{extension}") as logfile_gan:
    dictio_gan = json.load(logfile_gan)
plt.clf()
keys = np.array(sorted(list(map(float, dictio_arome.keys()))))
values = np.array([dictio_arome[str(key)] for key in keys])
plt.hist(x=keys, weights=values, density=True, bins=(v_max-v_min) * 2, range=(v_min, v_max), label="AROME", alpha=0.5)
keys = np.array(sorted(list(map(float, dictio_gan.keys()))))
values = np.array([dictio_gan[str(key)] for key in keys])
plt.hist(x=keys, weights=values, density=True, bins=(v_max-v_min) * 2, range=(v_min, v_max), label="GAN", alpha=0.5)
plt.xlabel("s_rr")
plt.ylabel(stat_name)
plt.legend()
plt.savefig(f"{path}MIX_{stat_name}_{v_min}_{v_max}.png")

stat_name = "order_0"
extension = ".npy"
map_gan = np.load(f"{path}~Set_{set_num}/~stats/{stat_name}{extension}")
map_list = [map_gan]
for idx_map, order_map in enumerate(map_list):
    plt.clf()
    fig, axes = plt.subplots()
    img = axes.imshow(order_map, origin="lower", vmin=0, vmax=0.5)
    plt.title(stat_name)
    fig.colorbar(img)
    fig.savefig(f"{save_dir}{stat_name}_{idx_map}.png")
    plt.close(fig)
stat_name = "order_1"
extension = ".npy"
map_arome = np.load(f"{path}AROME/{stat_name}{extension}")
map_gan = np.load(f"{path}GAN/{stat_name}{extension}")
map_list = [map_arome, map_gan]
for idx_map, order_map in enumerate(map_list):
    plt.clf()
    fig, axes = plt.subplots()
    img = axes.imshow(order_map, origin="lower", vmin=0, vmax=0.9)
    plt.title(stat_name)
    fig.colorbar(img)
    fig.savefig(f"{path}MIX_{stat_name}_{idx_map}.png")
    plt.close(fig)
"""
if comp:
    stat_name = "area_proportion"
    extension = ".json"
    plt.clf()
    with open(f"{path}AROME{'_IS' if arome_is else ''}/{stat_name}{extension}") as logfile_arome:
        dictio_arome = json.load(logfile_arome)
        dictio_arome = {float(key): value for key, value in dictio_arome.items()}
    keys = np.array(sorted(list(map(float, dictio_arome.keys()))))
    values = np.array([dictio_arome[key] for key in keys])
    plt.plot(keys, values, label=f"AROME{'_IS' if arome_is else ''}", color="purple")
    with open(f"{path}AROME/{stat_name}{extension}") as logfile_arome:
        dictio_arome = json.load(logfile_arome)
        dictio_arome = {float(key): value for key, value in dictio_arome.items()}
    keys = np.array(sorted(list(map(float, dictio_arome.keys()))))
    values = np.array([dictio_arome[key] for key in keys])
    plt.plot(keys, values, label="AROME", color="black")
    for set_num in set_list:
        with open(f"{path}GAN/Set_{set_num}/{stat_name}{extension}") as logfile_gan:
            dictio_gan = json.load(logfile_gan)
            dictio_gan = {float(key): value for key, value in dictio_gan.items()}
        keys = np.array(sorted(list(map(float, dictio_gan.keys()))))
        values = np.array([dictio_gan[key] for key in keys])
        plt.plot(keys, values, label=dict_caract_set[set_num], color=dict_color_set[set_num])
    plt.yscale("log")
    plt.xlabel("s_rr")
    plt.ylabel("Area proportion (log10)")
    plt.title(f"Area proportion for precipitation >= s_rr GANs vs AROME{' IS' if arome_is else ''}")
    plt.legend()
    plt.savefig(f"{save_dir}comp_area_proportion{'_IS' if arome_is else ''}.png")
else:
    stat_name = "area_proportion"
    extension = ".json"
    with open(f"{path}AROME{'_IS' if arome_is else ''}/{stat_name}{extension}") as logfile_arome:
        dictio_arome = json.load(logfile_arome)

    with open(f"{path}GAN/Set_{set_num}/{stat_name}{extension}") as logfile_gan:
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

stat_name = "extracted_values"
extension = ".json"

v_min, v_max = 0, 20

with open(f"{path}AROME{'_IS' if arome_is else ''}/{stat_name}{extension}") as logfile_arome:
    dictio_arome = json.load(logfile_arome)
with open(f"{path}GAN/Set_{set_num}/{stat_name}{extension}") as logfile_gan:
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

with open(f"{path}AROME{'_IS' if arome_is else ''}/{stat_name}{extension}") as logfile_arome:
    dictio_arome = json.load(logfile_arome)
with open(f"{path}GAN/Set_{set_num}/{stat_name}{extension}") as logfile_gan:
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

with open(f"{path}AROME{'_IS' if arome_is else ''}/{stat_name}{extension}") as logfile_arome:
    dictio_arome = json.load(logfile_arome)
with open(f"{path}GAN/Set_{set_num}/{stat_name}{extension}") as logfile_gan:
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
