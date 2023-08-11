#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:17:43 2022

@author: brochetc

metrics computation configuration tools

"""
import argparse
from evaluation_backend import var_dict
import numpy as np


def str2bool(v):
    return v.lower() in ('true')

def str2list(li):
    if type(li)==list:
        li2=li
        return li2
    elif type(li)==str:
        
        if ', ' in li :
            li2=li[1:-1].split(', ')
        else :
            
            li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))
    
        
def retrieve_domain_parameters(path, instance_num):
    
    with open(path+'ReadMe_'+str(instance_num)+'.txt', 'r') as f :
        print('opening ReadMe')
        li=f.readlines()
        for line in li:
            if "crop_indexes" in line :
                CI=[int(c) for c in str2list(line[15:-1])]
        
            if "var_names" in line :
                var_names = [v[1:-1] for v in str2list(line[12:-1])]
        print('variables', var_names)
        f.close()
        try :
            var_real_indices=[var_dict[v] for v in var_names]
        except NameError :
            raise NameError('Variable names not found in configuration file')
        
        try :
            print(CI)
        except UnboundLocalError :
            CI=[78,206,55,183]
        
    return CI, var_names

def getAndNameDirs(root_expe_path):
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--glob_name', type = str, help = 'Global experiment name', default = 'stylegan2_stylegan_')

    parser.add_argument('--expe_set', type = int, help = 'Set of experiments to dig in.', default = 1)
    parser.add_argument('--lr0', type = str2list, help = 'Set of initial learning rates', default = [0.002])
    parser.add_argument('--instance_num', type = str2list, help = 'Instances of experiment to dig in', default = [1])
    parser.add_argument('--conditional', type = str2bool, help ='Whether experiment is conditional', default = False)
    
    
    parser.add_argument('--ch_multip', type=int, help='channel multiplier', default=2)

    parser.add_argument('--step', type=int, default=147000, help="Which step to compute metrics on")
    parser.add_argument('--batch_sizes',type = str2list, default=[16], help = 'Set of batch sizes experimented')
    parser.add_argument('--latent_dim', type=str2list, default=[512], help='size of the latent vector')
    parser.add_argument("--dom_sizes", type=str2list, default = [256], help="size of domain")
    parser.add_argument('--variables', type = str2list, nargs="+", default=['u','v','t2m','z500','t850','tpw850'],
        help = 'List of subset of variables to compute metrics on') # provide as: --variables ['u','v'] ['t2m'] for instance (list after list)

    parser.add_argument("--use_noise", type=str2bool, default=[True], help="prevent noise injection if false")
    parser.add_argument("--mean_pert", type=str2bool, default = False, help="dataset with mean/pert separated")

    parser.add_argument('--step', type=int, default=147000, help="Which step to compute metrics on")
    parser.add_argument('--batch_sizes',type = str2list, default=[16], help = 'Set of batch sizes experimented')
    parser.add_argument('--latent_dim', type=str2list, default=[512], help='size of the latent vector')
    parser.add_argument("--dom_sizes", type=str2list, default = [256], help="size of domain")
    parser.add_argument('--variables', type = str2list, nargs="+", default=['u','v','t2m','z500','t850','tpw850'],
        help = 'List of subset of variables to compute metrics on') # provide as: --variables ['u','v'] ['t2m'] for instance (list after list)

    parser.add_argument("--use_noise", type=str2bool, default=[True], help="prevent noise injection if false")
    parser.add_argument("--mean_pert", type=str2bool, default = False, help="dataset with mean/pert separated")

    
    multi_config=parser.parse_args()
    
    #print("mc", multi_config.variables)
    names=[]
    short_names=[]
    list_steps=[]
    
    for lr in multi_config.lr0:
        for batch in multi_config.batch_sizes :
            for instance in multi_config.instance_num:
                for dim in multi_config.latent_dim:
                    for vars in multi_config.variables:
                        for n in multi_config.use_noise:
                            for dom_size in multi_config.dom_sizes:

                                name = root_expe_path\
                                        +multi_config.glob_name+f'dom_{dom_size}_lat-dim_'+str(dim)+'_bs_'+str(batch)\
                                        +'_'+str(lr)+'_'+str(lr)+'_ch-mul_'+str(multi_config.ch_multip)\
                                        + '_vars_' + '_'.join(str(var) for var in vars)\
                                        +f'_noise_{n}'\
                                        +("_mean_pert" if multi_config.mean_pert else "")\
                                        +f'/Instance_'+str(instance)
                                names.append(name)
                                
                                short_names.append('Instance_{}_Batch_{}_LR_{}_LAT_{}'.format(instance, batch,lr, multi_config.latent_dim))
                                
                                #list_steps.append([51000*i for i in range(12)])
                                list_steps.append([multi_config.step])
                                """
                                if int(batch)==0:
                                    list_steps.append([0])
                                
                                if int(batch)<=64 and int(batch)>0:
                                    list_steps.append([1500*k for k in range(40)]+[59999])
                                    
                                else:
                                    list_steps.append([1500*k for k in range(22)])
                                """
                    
    data_dir_names, log_dir_names = [f+'/samples/' for f in names],[f+'/log/' for f in names]
    
    multi_config.data_dir_names = data_dir_names
    multi_config.log_dir_names = log_dir_names
    multi_config.short_names = short_names
    multi_config.list_steps = list_steps
    
    multi_config.length = len(data_dir_names)
    
    return multi_config

def select_Config(multi_config, index):
    """
    Select the configuration of a multi_config object corresponding to the given index
    and return it in an autonomous Namespace object
    
    Inputs :
        multi_config : argparse.Namespace object as returned by getAndNameDirs
        
        index : int
    
    Returns :
        
        config : argparse.Namespace object
    
    """    
    
    insts = len(multi_config.instance_num)
    batches = len( multi_config.batch_sizes)
    lr0s = len(multi_config.lr0)
    
    
    config = argparse.Namespace() # building autonomous configuration
    
    config.data_dir_f = multi_config.data_dir_names[index]
    config.log_dir = multi_config.log_dir_names[index]
    config.steps = multi_config.list_steps[index]
    
    config.short_name = multi_config.short_names[index]
    instance_index = index%insts
    
    
    config.lr0 = multi_config.lr0[((index//insts)//batches)%lr0s]
    config.batch = multi_config.batch_sizes[((index//insts)%batches)]
    config.instance_num = multi_config.instance_num[instance_index]
    config.mean_pert = multi_config.mean_pert
    
    config.variables = multi_config.variables[index] if len(multi_config.variables) > 1 else \
                        multi_config.variables[0] ## assuming same subset of variables for each experiment, by construction
    #print("conf ", config.variables)    
    return config

class Experiment():
    """
    
    Define an "experiment" on the basis of the outputted config by select_Config
    
    This the base class to manipulate data from the experiment.
    
    It should be used exclusively as a set of informations easily regrouped in an abstract class.
    
    """
    def __init__(self, expe_config):
        
        self.data_dir_f = expe_config.data_dir_f
        self.log_dir = expe_config.log_dir
        self.expe_dir = self.log_dir[:-4]
            
        self.steps = expe_config.steps
        
        self.instance_num = expe_config.instance_num

        self.mean_pert = expe_config.mean_pert
        
        
        ###### variable indices selection : unchanged if subset is [], else selected
        
        indices = retrieve_domain_parameters(self.expe_dir, self.instance_num)
        
        self.CI, self.var_names = indices
        print("Crop indices: ", self.CI)

        self.dom_size = np.abs(self.CI[1]-self.CI[0])
        
        ########### Subset selection #######
        
        var_dict_fake = { v : i for i, v in enumerate(self.var_names)} # assuming variables are ordered !
        
        self.VI_f = list(var_dict_fake.values()) # warning, special object if not modified
        #print(set(expe_config.variables), expe_config.variables, set(self.var_names), self.var_names)        
        assert set(expe_config.variables) <= set(self.var_names)
        
        if set(expe_config.variables) != set(self.var_names) \
        and not len(expe_config.variables)==0 :            
            
            self.VI_f = [var_dict_fake[v] for v in expe_config.variables ]
            
        ##### final setting of variable indices
            
        self.VI = [var_dict[v] for v in expe_config.variables]
        
        
        
        self.var_names = expe_config.variables
        
        print('Indices of selected variables in samples (real/fake) :',
              self.VI, self.VI_f)
        
        assert len(self.VI)==len(self.VI_f) # sanity check
       
        
        
        
    def __print__(self) :
    
        print("Fake data directory {}".format(self.data_dir_f))
        print("Log directory {}".format(self.log_dir))
        print("Experiment directory {}".format(self.expe_dir))
        print("Instance num {}".format(self.instance_num))
        print("Step list : ", self.steps)
        print("Crop indices", self.CI)
        print("Var names" , self.var_names)
        print("Var indices", self.VI)
