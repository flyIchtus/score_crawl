#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:17:43 2022

@author: brochetc

metrics computation configuration tools

"""
import argparse
from evaluation_backend import var_dict


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
                print(CI)
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

def getAndNameDirs():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--root_expe_path', type = str, help = 'Root of dir expe', default = '/scratch/mrmn/moldovang/')

    parser.add_argument('--glob_name', type = str, help = 'Global experiment name', default = 'stylegan2_stylegan_512')

    parser.add_argument('--expe_set', type = int, help = 'Set of experiments to dig in.', default = 1)
    parser.add_argument('--lr0', type = str2list, help = 'Set of initial learning rates', default = [0.001,0.005])
    parser.add_argument('--batch_sizes',type = str2list, help = 'Set of batch sizes experimented', default=[8,16,32])
    parser.add_argument('--instance_num', type = str2list, help = 'Instances of experiment to dig in', default = [1,2,3,4])
    parser.add_argument('--variables', type = str2list, help = 'List of subset of variables to compute metrics on', default =[])
    parser.add_argument('--conditional', type = str2bool, help ='Whether experiment is conditional', default = False)
    parser.add_argument('--n_samples', type = int, help = 'Set of experiments to dig in.', default = 100)




    multi_config=parser.parse_args()



    N_samples=multi_config.n_samples
    names=[]
    short_names=[]
    list_steps=[]
    
    root_expe_path = multi_config.root_expe_path

    for lr in multi_config.lr0:
        for batch in multi_config.batch_sizes :
            for instance in multi_config.instance_num:
                
                names.append(root_expe_path+'Set_'+str(multi_config.expe_set)\
                                    +'/'+multi_config.glob_name+'_'+str(batch)\
                                    +'_'+str(lr)+'_'+str(lr)+'/Instance_'+str(instance))
                
                short_names.append('Instance_{}_Batch_{}_LR_{}'.format(instance, batch,lr))
                
                #list_steps.append([51000*i for i in range(12)])
                
                if int(batch)==0:
                    list_steps.append([0])
                
                if int(batch)<=64 and int(batch)>0:
                    list_steps.append([1500*k for k in range(40)]+[59999])
                    
                else:
                    list_steps.append([1500*k for k in range(22)])
                    
    data_dir_names, log_dir_names = [f+'/samples/' for f in names],[f+'/log/' for f in names]
    
    multi_config.data_dir_names = data_dir_names
    multi_config.log_dir_names = log_dir_names
    multi_config.short_names = short_names
    multi_config.list_steps = list_steps
    
    multi_config.length = len(data_dir_names)
    
    return multi_config, N_samples

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
    
    config.variables = multi_config.variables ## assuming same subset of variables for each experiment, by construction
        
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
        
        
        ###### variable indices selection : unchanged if subset is [], else selected
        
        indices = retrieve_domain_parameters(self.expe_dir, self.instance_num)
        
        self.CI, self.var_names = indices
        
        ########### Subset selection #######
        
        var_dict_fake = { v : i for i, v in enumerate(self.var_names)} # assuming variables are ordered !
        
        self.VI_f = list(var_dict_fake.values()) # warning, special object if not modified
                
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
