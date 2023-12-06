
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/moldovang/sxbigdata1/stylegan2-pytorch-master')

"""from utils import (

    real_fake,

    )"""


import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm# Collect all the font names available to matplotlib

#import plotly.figure_factory as ff

import matplotlib.gridspec as gridspec
import torch
from torch.nn import functional as F
import matplotlib.gridspec as gridspec

font_names = [f.name for f in fm.fontManager.ttflist]
#print(font_names)

mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2
#plt.rcParams['text.usetex'] = True




#print("sizes are", real.size(), fake.size())
# img_grid_real = torchvision.utils.make_grid(real[:8,:1,:,:],)
# img_grid_fake = torchvision.utils.make_grid(fake[:8,:1,:,:],)
# writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
# writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)



data_dir = '/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/eval_scores_cond/log/'

def denorm(data, Mean, Maxs, scale):

    return (1.0/scale) * Maxs * data + Mean

"""mse_SG = np.load('L_mse_' + str(0) + '_SG' + '.npy' )
mse_standard = np.load('L_mse_' + str(0) + '_standard' + '.npy' )
mse_Wplus = np.load('L_mse_' + str(0) + '_Wplus' + '.npy' )
avg_var = np.load('Avg_var_' + str(0) + '_SG' + '.npy' )
#mse_sgan = np.load('STYLE_GAN.npy')    
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0, 0, 1, 1])
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)

# Edit the major and minor ticks of the x and y axes
ax.xaxis.set_tick_params(which='major', size=8, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=5, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=8, width=1, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=5, width=1, direction='in', right='on')

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.plot(mse_SG, linewidth=1, color='black', label='StyleGAN')
ax.plot(mse_standard, linewidth=1, color='blue', label='AROME W')
ax.plot(mse_Wplus, linewidth=1, color='grey', label='AROME W+')
#ax.plot(mse_sgan, linewidth=1, color='blue', label='styleGAN2')
ax.set_yscale('log')

ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=14)
plt.savefig("mse_err"+".pdf", bbox_inches='tight')"""


###########################################################################################

for i in range(16):
    print("*"*30, i, "*"*30)
    per_cond = 7
    Ens_invert = np.load(data_dir + 'plot_invert_1000_raw/invertFsemble_2021-11-15_9_1000.npy')[i]
    Ens_normal = np.load(data_dir + 'plot_normal_full_raw/genFsemble_2021-11-15_9_1000.npy')[i * per_cond]
    Ens_normal_1 = np.load(data_dir + 'plot_normal_1_raw/genFsemble_2021-11-15_9_1000.npy')[i * per_cond]
    Ens_random = np.load(data_dir + 'plot_random_raw/genFsemble_2021-11-15_9_1000.npy')[i * per_cond]

    Ens_real = np.load(data_dir + 'plot_real_raw/Rsemble_2021-11-15_9.npy')[i]

    print(Ens_invert.mean(axis=(-2,-1)))
    print(Ens_normal.mean(axis=(-2,-1)))
    print(Ens_normal_1.mean(axis=(-2,-1)))
    print(Ens_random.mean(axis=(-2,-1)))
    print(Ens_real.mean(axis=(-2,-1)))

    Mean = np.load('/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/stat_files/Mean_4_var.npy')[1:].reshape(3,1,1)
    Maxs = np.load('/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/stat_files/MaxNew_4_var.npy')[1:].reshape(3,1,1)


    Ens_invert = denorm(Ens_invert, Mean, Maxs, 0.95)
    Ens_normal = denorm(Ens_normal, Mean, Maxs, 0.95)
    Ens_normal_1 = denorm(Ens_normal_1, Mean, Maxs, 0.95)
    Ens_random = denorm(Ens_random, Mean, Maxs, 0.95)
    Ens_real = denorm(Ens_real, Mean, Maxs, 0.95)

    print(Ens_invert.mean(axis=(-2,-1)))
    print(Ens_normal.mean(axis=(-2,-1)))
    print(Ens_normal_1.mean(axis=(-2,-1)))
    print(Ens_random.mean(axis=(-2,-1)))
    print(Ens_real.mean(axis=(-2,-1)))

    fig, axs = plt.subplots(3, 5, figsize=(10, 4.9))
        
    gs1 = gridspec.GridSpec(3, 5)
    gs1.update(wspace=0.175, hspace=0.175) # set the spacing between axes. 

    #plt.rcParams["figure.figsize"] = [50.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    axs = plt.subplot(gs1[0])
    c = axs.pcolor(Ens_invert[2,:,:], cmap="coolwarm",vmin = Ens_real[2,:,:].min(), vmax = Ens_real[2,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')   
    cb.ax.tick_params(labelsize=8)    
    cb.ax.yaxis.get_offset_text().set(size=8)    
    cb.outline.set_linewidth(1)              
    cb.update_ticks()


    axs = plt.subplot(gs1[1])
    c = axs.pcolor(Ens_normal[2,:,:], cmap="coolwarm",vmin = Ens_real[2,:,:].min(), vmax = Ens_real[2,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')   
    cb.ax.tick_params(labelsize=8)    
    cb.ax.yaxis.get_offset_text().set(size=8)    
    cb.outline.set_linewidth(1)              
    cb.update_ticks()

    axs = plt.subplot(gs1[2])
    c = axs.pcolor(Ens_normal_1[2,:,:], cmap="coolwarm",vmin = Ens_real[2,:,:].min(), vmax = Ens_real[2,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')   
    cb.ax.tick_params(labelsize=8)    
    cb.ax.yaxis.get_offset_text().set(size=8)    
    cb.outline.set_linewidth(1)              
    cb.update_ticks()


    axs = plt.subplot(gs1[3])
    c = axs.pcolor(Ens_random[2,:,:], cmap="coolwarm",vmin = Ens_real[2,:,:].min(), vmax = Ens_real[2,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')   
    cb.ax.tick_params(labelsize=8)    
    cb.ax.yaxis.get_offset_text().set(size=8)    
    cb.outline.set_linewidth(1)              
    cb.update_ticks()

    axs = plt.subplot(gs1[4])
    c = axs.pcolor(Ens_real[2,:,:], cmap="coolwarm", vmin = Ens_real[2,:,:].min(), vmax = Ens_real[2,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                           
    cb.update_ticks()

    ################################################

    axs = plt.subplot(gs1[5])
    c = axs.pcolor(Ens_invert[0,:,:], cmap="viridis", vmin = Ens_real[0,:,:].min(), vmax = Ens_real[0,:,:].max())
    axs.axis('off')

    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right') 
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                         
    cb.update_ticks()

    axs = plt.subplot(gs1[6])
    c = axs.pcolor(Ens_normal[0,:,:], cmap="viridis", vmin = Ens_real[0,:,:].min(), vmax = Ens_real[0,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right') 
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                         
    cb.update_ticks()

    axs = plt.subplot(gs1[7])
    c = axs.pcolor(Ens_normal_1[0,:,:], cmap="viridis", vmin = Ens_real[0,:,:].min(), vmax = Ens_real[0,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right') 
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                         
    cb.update_ticks()

    axs = plt.subplot(gs1[8])
    c = axs.pcolor(Ens_random[0,:,:], cmap="viridis", vmin = Ens_real[0,:,:].min(), vmax = Ens_real[0,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right') 
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                         
    cb.update_ticks()

    axs = plt.subplot(gs1[9])
    c = axs.pcolor(Ens_real[0,:,:], cmap="viridis", vmin = Ens_real[0,:,:].min(), vmax = Ens_real[0,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                          
    cb.update_ticks()

    ################################################

    axs = plt.subplot(gs1[10])
    c = axs.pcolor(Ens_invert[1,:,:], cmap="viridis", vmin = Ens_real[1,:,:].min(), vmax = Ens_real[1,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                          
    cb.update_ticks()

    axs = plt.subplot(gs1[11])
    c = axs.pcolor(Ens_normal[1,:,:], cmap="viridis", vmin = Ens_real[1,:,:].min(), vmax = Ens_real[1,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                          
    cb.update_ticks()

    axs = plt.subplot(gs1[12])
    c = axs.pcolor(Ens_normal_1[1,:,:], cmap="viridis", vmin = Ens_real[1,:,:].min(), vmax = Ens_real[1,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                          
    cb.update_ticks()

    axs = plt.subplot(gs1[13])
    c = axs.pcolor(Ens_random[1,:,:], cmap="viridis", vmin = Ens_real[1,:,:].min(), vmax = Ens_real[1,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                          
    cb.update_ticks()

    axs = plt.subplot(gs1[14])
    c = axs.pcolor(Ens_real[1,:,:], cmap="viridis", vmin = Ens_real[1,:,:].min(), vmax = Ens_real[1,:,:].max())
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)
    cb.outline.set_linewidth(1)                          
    cb.update_ticks()


    plt.savefig(f"comparison_real_gen_{i}.jpg",  dpi=600, transparent=False, bbox_inches='tight')
    plt.close()
#################################################################################################

fig, axs = plt.subplots(2, 3, figsize=(6, 3.15))

#plt.rcParams["figure.figsize"] = [50.00, 3.50]
plt.rcParams["figure.autolayout"] = True
vmin = 0.00001
vmax = 0.03
gs1 = gridspec.GridSpec(2, 3)
gs1.update(wspace=0.175, hspace=0.175) # set the spacing between axes. 



Dv_w=abs(Ens_real_var[1,:,:]-Ens_proj_var_w[1,:,:])
Du_w=abs(Ens_real_var[0,:,:]-Ens_proj_var_w[0,:,:])
Dt_w=abs(Ens_real_var[2,:,:]-Ens_proj_var_w[2,:,:])

Dv_wplus=abs(Ens_real_var[1,:,:]-Ens_proj_var_wplus[1,:,:])
Du_wplus=abs(Ens_real_var[0,:,:]-Ens_proj_var_wplus[0,:,:])
Dt_wplus=abs(Ens_real_var[2,:,:]-Ens_proj_var_wplus[2,:,:])

axs = plt.subplot(gs1[0])
c=axs.pcolor(Dt_w, cmap="Greys", vmax = vmax/1., vmin = vmin)
axs.set_title('t2m')
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()

axs = plt.subplot(gs1[1])
c=axs.pcolor(Du_w, cmap="Greys", vmax = vmax, vmin = vmin)
axs.set_title('u')
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()

axs = plt.subplot(gs1[2])
c=axs.pcolor(Dv_w, cmap="Greys", vmax = vmax, vmin = vmin)
axs.set_title('v')
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()

axs = plt.subplot(gs1[3])
c=axs.pcolor(Dt_wplus, cmap="Greys", vmax = vmax/1., vmin = vmin)
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()

axs = plt.subplot(gs1[4])
c=axs.pcolor(Du_wplus, cmap="Greys", vmax = vmax, vmin = vmin)
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()

axs = plt.subplot(gs1[5])
c=axs.pcolor(Dv_wplus, cmap="Greys", vmax = vmax, vmin = vmin)
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()


plt.savefig("compa_AE_proj.jpg",  dpi=600, transparent=False, bbox_inches='tight')





