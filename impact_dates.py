import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import random

expes = ['normal_full', 'normal_1', 'random_1']

data_dir = "/home/mrmn/brochetc/score_crawl/log/research_extremes/"
fake_dir = "/scratch/mrmn/moldovang/tests_CGAN/"

expes_dict = {"extrapolation" : ["extrapolation_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']/", "extrap", "genFsemble"],
        "inversion" : ['INVERSION_200/', 'invert', 'invertFsemble'],
        "random_1" : ["random_['1', '0', '0', '0', '0', '0', '1', '0', '1', '0', '1', '1', '0', '1']/", 'random_1', 'genFsemble'], 
        "random_2" : ["random_['0', '0', '0', '1', '0', '0', '0', '1', '1', '0', '1', '1', '0', '1']/", 'random_2', 'genFsemble'],
        "normal_full" : ["normal_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']/", 'normal_full', 'genFsemble'],
        "normal_1" : ["normal_['1', '1', '1', '1', '0', '0', '1', '1', '1', '0', '1', '1', '0', '1']/", 'normal_1', 'genFsemble'],
        "normal_spec" : ["normal_['1', '1', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '0']/", 'normal_spec', 'genFsemble'],
        }

offsets = [0,4,8]

#expes = ['random_1','random_1','random_1','random_1']
#expes = ['normal_1','normal_1','normal_1','normal_1']
#expes = ['normal_full','normal_full','normal_full','normal_full']
expes = ['random_2','random_2','random_2','random_2']
expes = ['normal_spec','normal_spec','normal_spec','normal_spec']

Means = np.load('/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/stat_files/Mean_4_var.npy')[1:4].reshape(1,3,1,1)
Maxs = np.load('/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/stat_files/MaxNew_4_var.npy')[1:4].reshape(1,3,1,1)

def denorm(data, Means, Maxs, scale):
    return (1.0/scale) * Maxs * data + Means

N_random = 5
per_cond = 112//16
for offset, expe in zip(offsets,expes) :

    sample_dir = fake_dir + expes_dict[expe][0] + 'samples/'
    real_dir = '/scratch/mrmn/moldovang/tests_CGAN/REAL_256/samples/'

    print(expe, offset)
    ids_top = np.load(data_dir + f'idx_top_{expe}_{offset}.npy')
    ids_bottom = np.load(data_dir + f'idx_bottom_{expe}_{offset}.npy')

    dates_top = [i[0] for i in ids_top]

    print(len(dates_top),len(set(dates_top)))

    dates_bottom = [i[0] for i in ids_bottom]
    print(len(dates_bottom), len(set(dates_bottom)))

    set_intersect = set(dates_bottom).intersection(set(dates_top))

    print('intersect', len(set_intersect))

    if offset==0:

        choices = random.sample(range(len(ids_top)), N_random)

        for sample_idx in choices:

            print(ids_top[sample_idx])

            date, lt, mb, count = ids_top[sample_idx]

            mb = int(mb)

            data_fake = np.load(sample_dir + f"genFsemble_{date}_{lt}_1000_16_112.npy")

            sample_fake = data_fake[mb]

            mb0 = per_cond * (mb//per_cond)

            if mb0!=mb:
                sample2 = data_fake[mb0]
            else:
                sample2 = data_fake[mb + 1]

            mean = data_fake[per_cond * (mb//per_cond) : per_cond * (mb//per_cond) + per_cond].mean(axis=0)

            sample_real = np.load(real_dir + f"Rsemble_{date}_{lt}.npy")[mb//per_cond].astype(np.float32)

            sample_unbiased = sample_fake - mean + sample_real

            sample2 = sample2 - mean + sample_real

            sample_fake = denorm(sample_fake, Means, Maxs, 0.95).squeeze()

            sample_real = denorm(sample_real, Means, Maxs, 0.95).squeeze()

            sample_unbiased = denorm(sample_unbiased, Means, Maxs, 0.95).squeeze()

            ##### plotting real v fake

            print(sample_real.shape, sample_fake.shape)

            fig, axs = plt.subplots(2,3,figsize = (6,6), sharex=True, sharey=True)

            axs[0,0].imshow(sample_real[0], origin = 'lower', cmap='viridis')

            axs[0,1].imshow(sample_real[1], origin = 'lower', cmap='viridis')

            axs[0,2].imshow(sample_real[2], origin = 'lower', cmap='coolwarm')

            axs[1,0].imshow(sample_fake[0], origin = 'lower', cmap='viridis', vmin = sample_real[0].min(), vmax = sample_real[0].max())

            axs[1,1].imshow(sample_fake[1], origin = 'lower', cmap='viridis', vmin = sample_real[1].min(), vmax = sample_real[1].max())

            axs[1,2].imshow(sample_fake[2], origin = 'lower', cmap='coolwarm', vmin = sample_real[2].min(), vmax = sample_real[2].max())

            fig.tight_layout()

            plt.savefig(data_dir + f'{expe}_real_v_fake_{date}_{lt}_{mb}_{mb//per_cond}_{offset}_top.png')

            plt.close()

            ##### plotting real v fake unbiased

            fig, axs = plt.subplots(2,3, figsize = (6,6), sharex=True, sharey=True)

            axs[0,0].imshow(sample_real[0], origin = 'lower', cmap='viridis')

            axs[0,1].imshow(sample_real[1], origin = 'lower', cmap='viridis')

            axs[0,2].imshow(sample_real[2], origin = 'lower', cmap='coolwarm')

            axs[1,0].imshow(sample_unbiased[0], origin = 'lower', cmap='viridis', vmin = sample_real[0].min(), vmax = sample_real[0].max())

            axs[1,1].imshow(sample_unbiased[1], origin = 'lower', cmap='viridis', vmin = sample_real[1].min(), vmax = sample_real[1].max())

            axs[1,2].imshow(sample_unbiased[2], origin = 'lower', cmap='coolwarm', vmin = sample_real[2].min(), vmax = sample_real[2].max())

            fig.tight_layout()

            plt.savefig(data_dir + f'{expe}_real_v_fake_unb_{date}_{lt}_{mb}_{mb//per_cond}_{offset}_top.png')

            plt.close()
        """
        choices = random.sample(len(ids_bottom), N_random)

        for sample_idx in choices:

            print(ids_bottom[sample_idx])

            date, lt, mb, count = ids_bottom[sample_idx]

            data_fake = np.load(sample_dir + f"genFsemble_{date}_{lt}_1000_16_112.npy")

            sample_fake = data_fake[mb]

            mb0 = per_cond * (mb//per_cond)

            if mb0!=mb:
                sample2 = data_fake[mb0]
            else:
                sample2 = data_fake[mb + 1]

            mean = data_fake[per_cond * (mb//per_cond) : per_cond * (mb//per_cond) + per_cond].mean(axis=0)

            sample_real = np.load(sample_dir + f"Rsemble_{date}_{lt}.npy")[mb//per_cond].astype(np.float32)

            sample_unbiased = sample_fake - mean + sample_real

            sample2 = sample2 - mean + sample_real

            sample_fake = denorm(sample_fake, Means, Maxs, 0.95)

            sample_real = denorm(sample_real, Means, Maxs, 0.95)

            ##### plotting real v fake

            fig, axs = plt.subplots(2,3,1, figsize = (6,6))

            axs[0] = plt.imshow(sample_real[0], origin = 'lower', cmap='viridis')

            axs[1] = plt.imshow(sample_real[1], origin = 'lower', cmap='viridis')

            axs[2] = plt.imshow(sample_real[2], origin = 'lower', cmap='coolwarm')

            axs[3] = plt.imshow(sample_fake[0], origin = 'lower', cmap='viridis', vmin = sample_real[0].min(), vmax = sample_real[0].max())

            axs[4] = plt.imshow(sample_fake[1], origin = 'lower', cmap='viridis', vmin = sample_real[1].min(), vmax = sample_real[1].max())

            axs[5] = plt.imshow(sample_fake[2], origin = 'lower', cmap='coolwarm', vmin = sample_real[2].min(), vmax = sample_real[2].max())

            plt.savefig(data_dir + f'{expe}_real_v_fake_{date}_{lt}_{mb}_{mb//per_cond}_{offset}_bottom.png'')

            plt.close()

            ##### plotting real v fake unbiased

            fig, axs = plt.subplots(2,3,1, figsize = (6,6))

            axs[0] = plt.imshow(sample_real[0], origin = 'lower', cmap='viridis')

            axs[1] = plt.imshow(sample_real[1], origin = 'lower', cmap='viridis')

            axs[2] = plt.imshow(sample_real[2], origin = 'lower', cmap='coolwarm')

            axs[3] = plt.imshow(sample_unbiased[0], origin = 'lower', cmap='viridis', vmin = sample_real[0].min(), vmax = sample_real[0].max())

            axs[4] = plt.imshow(sample_unbiased[1], origin = 'lower', cmap='viridis', vmin = sample_real[1].min(), vmax = sample_real[1].max())

            axs[5] = plt.imshow(sample_unbiased[2], origin = 'lower', cmap='coolwarm', vmin = sample_real[2].min(), vmax = sample_real[2].max())

            plt.savefig(data_dir + f'{expe}_real_v_fake_unb_{date}_{lt}_{mb}_{mb//per_cond}_{offset}_bottom.png'')

            plt.close()"""







