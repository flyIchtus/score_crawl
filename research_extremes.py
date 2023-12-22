import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

real_dir = '/scratch/mrmn/moldovang/tests_CGAN/REAL_256/'

fake_dir = "/scratch/mrmn/moldovang/tests_CGAN/" #random_['1', '0', '0', '0', '0', '0', '1', '0', '1', '0', '1', '1', '0', '1']/"

expes = {"extrapolation" : ["extrapolation_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']/", "extrap", "genFsemble"],
        "inversion" : ['INVERSION_200/', 'invert', 'invertFsemble'],
        "random_1" : ["random_['1', '0', '0', '0', '0', '0', '1', '0', '1', '0', '1', '1', '0', '1']/", 'random_1', 'genFsemble'], 
        "random_2" : ["random_['0', '0', '0', '1', '0', '0', '0', '1', '1', '0', '1', '1', '0', '1']/", 'random_2', 'genFsemble'],
        "normal_full" : ["normal_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']/", 'normal_full', 'genFsemble'],
        "normal_1" : ["normal_['1', '1', '1', '1', '0', '0', '1', '1', '1', '0', '1', '1', '0', '1']/", 'normal_1', 'genFsemble'],
        "normal_spec" : ["normal_['1', '1', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '0']/", 'normal_spec', 'genFsemble'],
        "real" : ["REAL_256/", "real", 'Rsemble'],        
        }


Means = np.load('/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/stat_files/Mean_4_var.npy')[1:4].reshape(1,3,1,1)
Maxs = np.load('/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/stat_files/MaxNew_4_var.npy')[1:4].reshape(1,3,1,1)

labels_file = '/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/large_lt/Large_lt_test_labels.csv'
parser = argparse.ArgumentParser()
parser.add_argument('--expe', type=str)
parser.add_argument('--unbiased', action='store_true')
parser.add_argument('--offset',type = int, default=0)

args = parser.parse_args()

expe = args.expe
offset = args.offset
print(offset)

data_dir = fake_dir + expes[expe][0] + 'samples/'
out_dir = './log/research_extremes/'
N_samples = 112
prefix = len("Rsemble_")

liste_dates = [d[prefix : prefix + 10] for d in os.listdir(real_dir+'samples/') if '_3.npy' in d]

num_dates = len(liste_dates)
liste_ech = [6,12,18,24,30,36,42]
num_ech = len(liste_ech)
#gen = np.zeros((N_samples * num_ech * num_dates,3,256,256))
"""for d_idx, dt in enumerate(liste_dates):
    print(dt)
    for ech_idx, ech in enumerate(liste_ech):
        
        gen[(d_idx * num_ech + ech_idx) * N_samples : (d_idx * num_ech + ech_idx + 1) * N_samples] = np.load(data_dir + f'genFsemble_{dt}_{ech}_1000_16_112.npy',mmap_mode='r').astype(np.float32)
"""
###
def denorm(data, Means, Maxs, scale):

    return (1.0/scale) * Maxs * data + Means

# Looking for extremes
#gen = denorm(gen, Means, Maxs, 0.95)
#Qs = np.percentile(gen, [0.001,99.999], axis=(0,2,3))
Qs = [np.array([-35.0,-35.0,243.0]), np.array([35.0,35.0,318.0])]
#print(Qs.shape, Qs)
count_top = 0
count_bottom = 0

list_bottom = []
list_top = []

var = ['u','v','t2m']
colors = ['viridis', 'viridis', 'coolwarm']
mins = [-35.0,-35.0,253.0]
maxs = [35.0,35.0,313.0]


idx_bottom = []
idx_top = []


for d_idx, dt in enumerate(liste_dates):
    print(dt)
    for ech_idx, ech in enumerate(liste_ech):
        sample0 =  denorm(np.load(data_dir + f'{expes[expe][-1]}_{dt}_{ech}_1000_16_112.npy',mmap_mode='r').astype(np.float32), Means, Maxs, 0.95)
        for mb_idx in range(N_samples):
            sample = sample0[mb_idx]
            s_idx = (d_idx * num_ech + ech_idx) * N_samples + mb_idx
            if offset>0:
                inf = ((sample[:,offset:-offset,offset:-offset] - Qs[0].reshape(3,1,1))<=0.0)
            else:
                inf = ((sample - Qs[0].reshape(3,1,1))<=0.0)            
            
            if np.any(inf):
                print("inf", inf.shape, np.sum(inf))
                list_bottom.append(s_idx)
                count_bottom+=1
                idx_bottom.append((dt, ech,mb_idx,np.sum(inf)))
                if count_bottom<=10:
                    for var_idx in range(3):
                        plt.imshow(sample[var_idx], origin='lower',cmap=colors[var_idx], vmin=mins[var_idx], vmax=maxs[var_idx])
                        plt.colorbar()
                        plt.savefig(out_dir + f'sample_bottom_{dt}_{ech}_{var[var_idx]}_{mb_idx}_{expe}.png')
                        plt.close()

            if offset>0:
                sup = ((sample[:,offset:-offset,offset:-offset] - Qs[1].reshape(3,1,1))>=0.0)
            else:
                sup = ((sample - Qs[1].reshape(3,1,1))>=0.0) 
            if np.any(sup):
                print("sup", sup.shape, np.sum(sup))
                list_top.append(s_idx)
                count_top+=1
                idx_top.append((dt, ech,mb_idx,np.sum(sup)))
                if count_top<=10:
                    for var_idx in range(3):
                        plt.imshow(sample[var_idx], origin='lower',cmap=colors[var_idx], vmin=mins[var_idx], vmax=maxs[var_idx])
                        plt.colorbar()
                        plt.savefig(out_dir + f'sample_top_{dt}_{ech}_{var[var_idx]}_{mb_idx}_{expe}.png')
                        plt.close()

print(count_top, count_bottom)
np.save(out_dir + f"idx_top_{expe}_{offset}.npy",idx_top)
np.save(out_dir + f"idx_bottom_{expe}_{offset}.npy",idx_bottom)