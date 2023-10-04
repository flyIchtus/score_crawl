import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
# import multivariate as mlt
from mpl_toolkits.axes_grid1 import ImageGrid
from metrics4arome import multivariate as mlt
plt.rcParams.update({'font.size': 30})


def str2list(li):
    if type(li) == list:
        li2 = li
        return li2
    elif type(li) == str:

        if ', ' in li:
            li2 = li[1:-1].split(', ')
        else:

            li2 = li[1:-1].split(',')
        return li2

    else:
        raise ValueError(
            "li argument must be a string or a list, not '{}'".format(type(li)))


parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, help = 'Global Path', default =  '/scratch/mrmn/gandonb/Exp_StyleGAN/Set_3/')
parser.add_argument('--distance', type = str, help = 'distance', default = '/Instance_1/log/test_distance_distance_metrics_step_6000_16000_26000_36000_44000_50000_16384.p')
parser.add_argument('--standalone', type = str,  default = '/Instance_1/log/test_standalone_standalone_metrics_step_6000_16000_26000_36000_44000_50000_16384.p')
parser.add_argument('--n_tests', type = int,  default = 1)
parser.add_argument('--var_names', type = str2list,  default = ['rr'])
parser.add_argument('--list_names', type = str2list,  default = ['stylegan2_stylegan_dom_256_lat-dim_512_bs_32_0.0002_0.0002_ch-mul_2_vars_rr_noise_True'])
opt = parser.parse_args()

l_steps = [6000, 16000, 26000, 36000, 44000, 50000]

Path = opt.path

#Distance = '/Instance_1/log/cond0_eval_distance_metrics_18944_18944_18944_18944_18944_18944_18944_18944_18944_18944.p'
#Distance = '/Instance_1/log/cond0_eval_distance_metrics_9472_9472_9472_9472_9472_9472_9472_9472_9472_9472.p'
Distance = opt.distance  #     '/Instance_1/log/test_distance_distance_metrics_100.p'
#Standalone = '/Instance_1/log/cond0_eval_standalone_metrics_9472_9472_9472_9472_9472_9472_9472_9472_9472_9472.p'
#Standalone = '/Instance_1/log/cond0_eval_standalone_metrics_18944_18944_18944_18944_18944_18944_18944_18944_18944_18944.p'
Standalone = opt.standalone  # '/Instance_1/log/test_standalone_standalone_metrics_100.p'

N_tests = opt.n_tests
var_names = opt.var_names  # ['u', 'v', 't2m']
N_vars = len(var_names)

list_names = opt.list_names  #  ['AROME_RED_0_0.001_0.001']
Length_list_names = len(list_names)

#res=pickle.load(open('cond0_eval_distance_metrics_9472_9472_9472_9472_9472_9472_9472_9472_9472_9472.p', 'rb'))
    
#stand_alone=pickle.load(open('cond0_eval_standalone_metrics_9472_9472_9472_9472_9472_9472_9472_9472_9472_9472.p', 'rb'))


# list_names = ['latent_space_0_0.001_0.001', 'real_0_0.001_0.001', 'interp_0.5_0_0.001_0.001',
#               'interp_1.25_0_0.001_0.001', 'interp_m0.25_0_0.001_0.001', 'noise_4x4_150_0_0.001_0.001',
#               'noise_8x8_150_0_0.001_0.001', 'noise_16x16_150_0_0.001_0.001',
#               'sm_0_0_0.001_0.001', 'sm_1_0_0.001_0.001', 'sm_2_0_0.001_0.001', 'sm_3_0_0.001_0.001',
#               'sm_4_0_0.001_0.001', 'sm_6_0_0.001_0.001', 'sm_8_0_0.001_0.001',
#               'sm_0_12_0_0.001_0.001', 'sm_1_12_0_0.001_0.001', 'sm_2_12_0_0.001_0.001',
#               'sm_3_12_0_0.001_0.001', 'sm_4_12_0_0.001_0.001', 'sm_6_12_0_0.001_0.001',
#               'sm_0_12_W_0_0.001_0.001', 'sm_4_12_W_0_0.001_0.001', 'sm_6_12_W_0_0.001_0.001',
#               'sm_8_8_0_0.001_0.001']
# list_names = ['latent_space_0_0.001_0.001', 'real_0_0.001_0.001', 'interp_0.5_0_0.001_0.001',
#               'interp_1.25_0_0.001_0.001', 'interp_m0.25_0_0.001_0.001', 'noise_4x4_150_0_0.001_0.001',
#               'noise_8x8_150_0_0.001_0.001', 'noise_16x16_150_0_0.001_0.001',
#               'sm_0_1_0_0.001_0.001', 'sm_1_2_0_0.001_0.001', 'sm_2_3_0_0.001_0.001', 'sm_3_4_0_0.001_0.001',
#               'sm_4_5_0_0.001_0.001', 'sm_6_7_0_0.001_0.001', 'sm_8_9_0_0.001_0.001',
#               'sm_0_12_0_0.001_0.001', 'sm_1_12_0_0.001_0.001', 'sm_2_12_0_0.001_0.001',
#               'sm_3_12_0_0.001_0.001', 'sm_4_12_0_0.001_0.001', 'sm_5_12_0_0.001_0.001', 'sm_6_12_0_0.001_0.001',
#               'sm_8_12_0_0.001_0.001',
#               'sm_0_12_W_0_0.001_0.001', 'sm_2_12_W_0_0.001_0.001', 'sm_4_12_W_0_0.001_0.001',
#               'sm_6_12_W_0_0.001_0.001', 'sm_8_12_W_0_0.001_0.001'

#               ]

SWD_scores = np.zeros((Length_list_names, len(l_steps), 5))
Multivar_scores = np.zeros((Length_list_names, len(l_steps), 2, N_vars, 100, 100))
for i in range(Length_list_names):
    # print(list_names[i])
    d_score = pickle.load(open(Path + list_names[i] + Distance, 'rb'))
    s_score = pickle.load(open(Path + list_names[i] + Standalone, 'rb'))
    print("N_tests", N_tests)
    print(d_score.keys())
    for k, j in enumerate(l_steps):
        SWD_scores[i, k] = d_score[j]['SWD_metric_torch'][0]

####################################### " PLOTING W1 MAP############################


# for i in range(Length_list_names):
#     s_score = pickle.load(open(Path + list_names[i] + Distance, 'rb'))
#     PW = s_score[36000]['pw_W1']

#     fig = plt.figure(figsize=(12, 3*N_vars))

#     grid = ImageGrid(fig, 111,
#                      nrows_ncols=(1, N_vars),
#                      axes_pad=(0.35, 0.35),
#                      label_mode="L",
#                      share_all=True,
#                      cbar_location="bottom",
#                      cbar_mode="edge",
#                      cbar_size="10%",
#                      cbar_pad="20%",
#                      )

#     axes = []

#     for j in range(N_vars):

#         ax = grid[j]
#         axes.append(ax)

#         if N_vars ==1:
#             im = ax.imshow(1e3*PW[:,:], origin ='lower', cmap = 'coolwarm')
#         else : 
#             im = ax.imshow(1e3*PW[j,:,:], origin ='lower', cmap = 'coolwarm')

#         cb = ax.cax.colorbar(im)

#     fig.tight_layout()
#     plt.savefig('W_map/W1_map_comparison_'+list_names[i]+'.png')
#     plt.close()


# Plotting spectra

# Spectrum_scores = np.zeros((Length_list_names, N_tests, N_vars, 90))
# Error_spec = np.zeros((Length_list_names, 3))
# for i in range(Length_list_names):

#     s_score = pickle.load(open(Path + list_names[i] + Standalone, 'rb'))
#     plt.rcParams.update({'font.size': 30})
#     N_vars = len(var_names)
#     for k, j in enumerate(l_steps):

#         Spectrum_scores[i, k] = s_score['spectral_compute']

#     fake_spectrum = np.mean(Spectrum_scores[i], axis=0)
#     Scale = np.log10(np.linspace(2.6, 117.0, 45))

#     res_spectrum_real = pickle.load(open('real3stand_alone_metrics_66048.p', 'rb'))[
#         'spectral_compute']
#     # res_spectrum_real = pickle.load(open(path+'spectral_distrib_standalone_metrics_66048.p', 'rb'))['spectral_distrib']
#     real_spectrum = res_spectrum_real.squeeze()

#     for j in range(N_vars):

#         fig = plt.figure(figsize=(18, 15))

#         lab = 'Mean GAN Spectrum'

#         plt.plot(Scale, np.log10(
#             fake_spectrum[j, :]), 'r-', linewidth=2.5, label=lab)

#         plt.grid()

#         plt.xticks([0.5, 1.0, 2.0], [
#                    r'$10^{2}$', r'$10$', r'$10^{0.5}$'], fontsize='28')
#         plt.yticks(fontsize='28')
#         plt.xlabel(r'Spatial Scale ($km$)', fontsize='32', fontweight='bold')

#         plt.plot(Scale, np.log10(
#             real_spectrum[j, :, 0]), 'k-', linewidth=2.5, label='Mean AROME Spectrum')
#         plt.plot(Scale, np.log10(
#             real_spectrum[j, :, 1]), 'k--', linewidth=2.5, label='Q10-90 AROME-EPS')
#         plt.plot(Scale, np.log10(
#             real_spectrum[j, :, 2]), 'k--', linewidth=2.5, label='Q10-90 AROME-EPS')

#         fig.tight_layout()
#         plt.savefig(
#             'spectral/Spectral_PSD_{}'.format(var_names[j]) + list_names[i]+'.png')
#         plt.close()

#         spec = fake_spectrum[j]
#         spec0_real = real_spectrum[j, :, 0]
#         psd_rmse = 10 * \
#             np.sqrt(np.mean(((np.log10(spec)-np.log10(spec0_real))**2)))
#         # print('PSD error {} in dB'.format(var_names[j]), psd_rmse)
#         Error_spec[i, j] = psd_rmse


# W1 scores

W1_scores = np.zeros((Length_list_names, len(l_steps)))
W1_scores_c = np.zeros((Length_list_names, len(l_steps)))
for i in range(Length_list_names):
    d_score = pickle.load(open(Path + list_names[i] + Distance, 'rb'))
    for k, j in enumerate(l_steps):
        W1_scores[i, k] = d_score[j]['W1_random_NUMPY'][0]
        W1_scores_c[i, k] = d_score[j]['W1_Center_NUMPY'][0]


# CORRELATION LENGTH
# fake_corr_length_scores = np.zeros(
#     (Length_list_names, N_tests, N_vars, 127, 127))
# mae_corr = np.zeros((Length_list_names))
# std_ae_corr = np.zeros((Length_list_names))
# max_ae_corr = np.zeros((Length_list_names))
# for i in range(Length_list_names):
#     s_score = pickle.load(open(Path + list_names[i] + Standalone, 'rb'))
#     for j in range(N_tests):
#         fake_corr_length_scores[i, j] = s_score[j]['ls_metric']
#     fake_corr_length = np.mean(fake_corr_length_scores[i], axis=0)
#     # print(fake_corr_length.shape)

#     real_corr = np.load('real_corr_length.npy')

#     N_vars = len(var_names)

#     fig = plt.figure(figsize=(12, 3*N_vars))

#     grid = ImageGrid(fig, 111,
#                     nrows_ncols=(2, N_vars),
#                     axes_pad=(0.35, 0.35), 
#                     label_mode="L",
#                     share_all=True,
#                     cbar_location="bottom",
#                     cbar_mode="edge",
#                     cbar_size="10%",
#                     cbar_pad="20%",
#                     )
    
#     for k in range(N_vars) : 
    
#         ax4 = grid[N_vars+k]
#         im4 = ax4.imshow(fake_corr_length[k, :, :], origin='lower')

#         ax = grid[k]
#         ax.imshow(real_corr[k, :, :], origin='lower', vmin=fake_corr_length[k].min(
#         ), vmax=fake_corr_length[k].max())

#         cb4 = ax4.cax.colorbar(im4)

#     fig.tight_layout()
#     plt.savefig('corr_length/Length_scales_map_comparison' +
#                 list_names[i]+'.png')
#     plt.close()
    
#     ######### computing statistics
    
#     diff_corr = real_corr[:N_vars] - fake_corr_length
    
#     mae_corr[i] = np.abs(diff_corr).mean()
#     std_ae_corr[i] = np.abs(diff_corr).std()
#     max_ae_corr[i] = np.abs(diff_corr).max()
###############################################"FINISHED CORR LENGTH###########################    
if N_vars>2 :
    for i in range(Length_list_names):
        
        #print("SWD scores for test", list_names[i], " are: ", np.mean(SWD_scores[i], axis=0))
        d_score = pickle.load(open(Path + list_names[i] + Distance, 'rb'))
        plt.rcParams.update({'font.size': 12})
        
        for k, j in enumerate(l_steps):
    
            Multivar_scores[i,k] = d_score[j]['multivar'][0].squeeze()
        RES = np.mean(Multivar_scores[i], axis=0)
        #print(RES.shape)
        data_r,data_f = RES[0], RES[1]
        #print(data_r.shape, data_f.shape)
    
    
        levels = mlt.define_levels(data_r,5)  #defining color and contour levels
    
        ncouples2 = data_f.shape[0]*(data_f.shape[0]-1)
    
        bins = np.linspace(tuple([-1 for i in range(ncouples2)]), tuple([1 for i in range(ncouples2)]),101, axis=1)
    
        var_r = (np.log(data_r), bins)
        var_f = (np.log(data_f), bins)
    
        mlt.plot2D_histo(var_f, var_r, levels, output_dir = 'multivariate/'+list_names[i], add_name='')
    
Results = np.zeros((Length_list_names, 13))
for i in range(Length_list_names):
    swd_averages = np.round(np.mean(SWD_scores[i], axis=0), 2)
    print(list_names[i] + ': ', np.round(np.mean(SWD_scores[i], axis=0), 2), np.round(np.mean(W1_scores[i], axis=0), 2),
           np.round(np.mean(W1_scores_c[i], axis=0), 2)) # np.round(mae_corr[i], 2), np.round(
        #       std_ae_corr[i], 2), np.round(max_ae_corr[i], 2), np.round(Error_spec[i, 0], 2),
        #   np.round(Error_spec[i, 1], 2), np.round(Error_spec[i, 2], 2))
    Results[i, 0] = swd_averages[0]
    Results[i, 1] = swd_averages[1]
    Results[i, 2] = swd_averages[2]
    Results[i, 3] = swd_averages[3]
    Results[i, 4] = swd_averages[4]
    # Results[i, 5] = np.round(np.mean(W1_scores[i], axis=0), 2)
    # Results[i, 6] = np.round(np.mean(W1_scores_c[i], axis=0), 2)
    # Results[i, 7] = np.round(mae_corr[i], 2)
    # Results[i, 8] = np.round(std_ae_corr[i], 2)
    # Results[i, 9] = np.round(max_ae_corr[i], 2)
    # Results[i, 10] = np.round(Error_spec[i, 0], 2)
    # Results[i, 11] = np.round(Error_spec[i, 1], 2)
    # Results[i, 12] = np.round(Error_spec[i, 2], 2)

np.savetxt("Results_intelligent_m.csv", Results, delimiter=",", fmt='%10.2f')

np.save('Results_intelligent_m.npy', Results)
