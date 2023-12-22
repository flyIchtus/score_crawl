import numpy as np
import os
import metrics4arome as METR
import pandas as pd
import pickle
import sys
import argparse

ceiling = 16384

real_dir = '/scratch/mrmn/moldovang/tests_CGAN/REAL_256/'

fake_dir = "/scratch/mrmn/moldovang/tests_CGAN/" #random_['1', '0', '0', '0', '0', '0', '1', '0', '1', '0', '1', '1', '0', '1']/"


vars_wo_orog = ['u', 'v', 't2m']

labels_file = '/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/large_lt/Large_lt_test_labels.csv'


expes = {"extrapolation" : ["extrapolation_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']/", "extrap", "genFsemble"],
        "inversion" : ['INVERSION_200/', 'invert', 'invertFsemble'],
        "random_1" : ["random_['1', '0', '0', '0', '0', '0', '1', '0', '1', '0', '1', '1', '0', '1']/", 'random_1', 'genFsemble'], 
        "random_2" : ["random_['0', '0', '0', '1', '0', '0', '0', '1', '1', '0', '1', '1', '0', '1']/", 'random_2', 'genFsemble'],
        "normal_full" : ["normal_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']/", 'normal_full', 'genFsemble'],
        "normal_1" : ["normal_['1', '1', '1', '1', '0', '0', '1', '1', '1', '0', '1', '1', '0', '1']/", 'normal_1', 'genFsemble'],
        "normal_spec" : ["normal_['1', '1', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '0']/", 'normal_spec', 'genFsemble'],
        }


Means = np.load('/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/stat_files/Mean_4_var.npy')[1:4].reshape(1,3,1,1)
Maxs = np.load('/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/stat_files/MaxNew_4_var.npy')[1:4].reshape(1,3,1,1)

Maxs_tot = np.load('/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/stat_files/max_rr.npy')[1:4]
Mins_tot = np.load('/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/stat_files/min_rr.npy')[1:4]




parser = argparse.ArgumentParser()
parser.add_argument('--expe', type=str)
parser.add_argument('--unbiased', action='store_true')
parser.add_argument('--offset', type=int, default=0)

args = parser.parse_args()

expe = args.expe
offset = args.offset
print(offset)
multivar_rescale = METR.metric2D('Multivariate data', lambda real,fake :  METR.multivariate.multi_variate_correlations(real, fake, Mins = Mins_tot, Maxs = Maxs_tot, density=True, offset=offset),\
                  vars_wo_orog, names=['Corr_f','Corr_r'])


fake_dir = fake_dir + expes[expe][0]
fake_prefix = expes[expe][-1]
log_prefix = expes[expe][1]


prefix = len("Rsemble_")

dates = [d[prefix : prefix + 10] for d in os.listdir(real_dir+'samples/') if '_3.npy' in d]
print(len(dates))
#dates = [d for d in dates if d[:10] not in ['2021-08-14','2021-10-13', '2021-10-18']]

standalone_metrics_list = ["spectral_compute", "ls_metric", "quant_map"]
distance_metrics_list = ["W1_random_NUMPY", "W1_Center_NUMPY", "SWD_metric_torch"]
#distance_metrics_list = ["multivar_rescale"]

#leadtimes = [int(sys.argv[1])]
leadtimes = [6,12,18,24,30,36,42]
print('leadtimes', leadtimes)
num_lt = len(leadtimes)

ensemble_members = 16

N_samples = min(ceiling, ensemble_members * len(leadtimes) * len(dates))

print('N_samples', N_samples)


def denorm(data, Means, Maxs, scale):
    return (1.0/scale) * Maxs * data + Means


fakes = []
real = []
if args.unbiased:
    fakes_unbiased = []

glob_idx = 0
for date_idx, date in enumerate(dates):
    for lt_idx, lt in enumerate(leadtimes):
        glob_idx = date_idx * num_lt + lt_idx
        if (glob_idx%50)==0:
            print(glob_idx, lt, date)

        if ensemble_members * (glob_idx + 1) <= N_samples:

            try:
                end = 16 if expe=='inversion' else 112
                step = 1 if expe=='inversion' else 112//(ensemble_members)

                fakes_all = np.load(fake_dir + f"samples/{fake_prefix}_{date}_{lt}_1000_16_112.npy", mmap_mode = 'r').astype(np.float32)

                fakes.append(fakes_all[0:end:step]) #extracting
                if args.unbiased:
                    fakes_mean = np.zeros((ensemble_members,3,256,256))
                    for i in range(ensemble_members):
                        fakes_mean[i] = fakes_all[i * step : (i + 1) * step].mean(axis=0)

                real.append(np.load(real_dir + f"samples/Rsemble_{date}_{lt}.npy", mmap_mode = 'r').astype(np.float32)[:ensemble_members])
                if args.unbiased : 
                    fakes_unbiased.append(fakes[-1] - fakes_mean + real[-1].mean(axis=0))

            except FileNotFoundError as err:
                print(err)
                pass
        else:
            break

print(len(fakes), len(real))
if args.unbiased: print(len(fakes_unbiased))

fakes = np.concatenate(fakes)
print(fakes.shape)
print(fakes[:,2,:,:].mean(axis=(0,-2,-1)))
real = np.concatenate(real)
print(real[:,2,:,:].mean(axis=(0,-2,-1)))

if args.unbiased:
    fakes_unbiased = np.concatenate(fakes_unbiased)


dic_res_real = {}
dic_res = {}
dic_res_unbiased = {}


for metr_name in distance_metrics_list:
    print(metr_name)
    

    if metr_name=='multivar_rescale':

        real_sc = denorm(real, Means,Maxs, 0.95)
        fakes_sc = denorm(fakes, Means, Maxs, 0.95)
        print(real_sc.min(), real_sc.max())
        print(fakes_sc.min(), fakes_sc.max())
        if not args.unbiased:
            dic_res[metr_name] = multivar_rescale(real_sc, fakes_sc)

        if args.unbiased:
            unb_sc = denorm(fakes_unbiased, Means, Maxs, 0.95)

            dic_res_unbiased[metr_name] = multivar_rescale(real_sc, unb_sc)
    else:
        if not args.unbiased:
            metric = getattr(METR, metr_name)
            dic_res[metr_name] = metric(real, fakes)

        if args.unbiased:
            metric = getattr(METR, metr_name)
            dic_res_unbiased[metr_name] = metric(real,fakes_unbiased)

lt0 = 0 if num_lt > 1 else lt

if 'multivar' in distance_metrics_list:
    distance_metrcsi_list[0] = 'multivar_scaled'

name_real = os.path.dirname(os.path.realpath(__file__)) +  f"/log/{log_prefix}_METR_" + '_'.join(distance_metrics_list) + f'_distance_metrics_{N_samples}_{lt0}_real_{offset}.p'
name_fake = os.path.dirname(os.path.realpath(__file__)) +  f"/log/{log_prefix}_METR_" + '_'.join(distance_metrics_list) + f'_distance_metrics_{N_samples}_{lt0}_raw_{offset}.p'
name_fake_unbiased = os.path.dirname(os.path.realpath(__file__)) +  f"/log/{log_prefix}_METR_" + '_'.join(distance_metrics_list) + f'_distance_metrics_{N_samples}_{lt0}_unbiased_{offset}.p'

#with open(name_real,'wb') as f:
#    pickle.dump(dic_res_real, f)
if not args.unbiased:
    with open(name_fake,'wb') as f:
        pickle.dump(dic_res, f)

if args.unbiased:
    with open(name_fake_unbiased,'wb') as f:
        pickle.dump(dic_res_unbiased, f)


"""for metr_name in standalone_metrics_list:
    print(metr_name)
    metric = getattr(METR, metr_name)

    dic_res[metr_name] = metric(fakes)
    dic_res_real[metr_name] = metric(real)
    dic_res_unbiased[metr_name] = metric(fakes_unbiased)

lt0 = 0 if num_lt > 1 else lt

name_real = os.path.dirname(os.path.realpath(__file__)) +  "/log/METR_" + '_'.join(standalone_metrics_list) + f'_standalone_metrics_{N_samples}_{lt0}_real.p'
name_fake = os.path.dirname(os.path.realpath(__file__)) +  "/log/METR_" + '_'.join(standalone_metrics_list) + f'_standalone_metrics_{N_samples}_{lt0}_raw.p'
name_fake_unbiased = os.path.dirname(os.path.realpath(__file__)) +  "/log/METR_" + '_'.join(standalone_metrics_list) + f'_standalone_metrics_{N_samples}_{lt0}_unbiased.p'

with open(name_real,'wb') as f:
    pickle.dump(dic_res_real, f)

with open(name_fake,'wb') as f:
    pickle.dump(dic_res, f)

with open(name_fake_unbiased,'wb') as f:
    pickle.dump(dic_res_unbiased, f)"""

