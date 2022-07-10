# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 19:51:15 2022

@author: Eidos
"""

import argparse
import numpy as np
import time
import os
import sys

# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if not top_path in sys.path:
    sys.path.append(top_path)
    
from experiment.attack.mid_runner_eval_attack import Mid_runner_eval
from toolbox.name_set import drone_set
from toolbox.name_set import name_set_csv
from toolbox.name_set import name_set_drone

def main(args):
    # The settign of the mfcc
    args.mfcc = {}
    args.mfcc['num_filter'] = 201
    args.mfcc['num_cep'] = 201
    args.mfcc['winlen'] = 1
    args.mfcc['winstep'] = 0.5
    args.mfcc['fs'] = 44100
    # Invalid for pkl mode (just for now)
    args.mfcc['mfcc_d1_switch'] = False
    args.mfcc['mfcc_d2_switch'] = False
    
    # All valid key
    name_set_list = ['prefix','date','drone_No','state','distance','index','suffix']
    # Predefine the key of the dic
    args.dic_choose = dict([(k,[]) for k in name_set_list])
    args.dic_aban = dict([(k,[]) for k in name_set_list])
    
    args.dic_choose["date"] = ['_20220304_', '_20220307_', '_20220312_', '_20220318_', '_20220319_',
                              '_20220327_', '_20220328_', '_20220329_', '_20220330_', '_20220331_', 
                              '_20220401_', '_20220402_', '_20220403_', '_20220404_', '_20220405_']
    # args.dic_choose["date"] = ['_20220227_']
    # args.dic_choose["drone_No"] = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_','_d7_','_d8_',
    #                               '_d9_','_d10_','_d11_','_d12_','_d13_','_d14_','_d15_','_d16_',
    #                               '_d17_','_d18_','_d19_','_d20_','_d21_','_d22_','_d23_','_d24_']
    # The background type used in training
    args.bg_type = ['_d1_', '_d6_', '_d7_', '_d13_', '_d15_', '_d16_', '_d20_', '_d22_']
    # The drone type used as attack drone in evaluation
    args.attack_type =  ['_d2_', '_d3_', '_d4_', '_d8_', '_d11_', '_d14_', '_d19_', '_d24_']
    
    # The drone type used in evaluation
    args.dic_choose["drone_No"] = list(set(name_set_drone["drone_No"]).difference(set(args.bg_type)))
    args.dic_choose["drone_No"].sort(key=name_set_drone["drone_No"].index)
    # The drone type used as register dorne in evaluation
    args.reg_type = list(set(args.dic_choose["drone_No"]).difference(set(args.attack_type)))
    args.reg_type.sort(key=name_set_drone["drone_No"].index)
    
    print(args.reg_type)
    
    # Path to find stored data
    args.originData_path = r'E:\1_Research\3_UAV_2\2_data\2_new_data'
    # Path to store trained model
    args.model_path = r'E:\1_Research\3_UAV_2\2_data\10_pkl_attack\10'
    
    args.pkl_savePath = r'E:\1_Research\3_UAV_2\2_data\10_pkl_attack'
    args.pkl_fileName = r'_201nf_201nc_1.00wl_0.50ws_8000lim_noise_24d.pkl'
    
    model_list = ['_QDA_', '_LDA_', '_LSVM_', '_SVM_', '_KNN_', '_DT_', '_RF_', '_GNB_']
    # args.model_name = r'_QDA_201nf_201nc_1.00wl_0.50ws_attack.m'
    # The drone dic
    time_start = time.time()
    
    for model in model_list:
        print('******%s******'%model)
        args.model_name = '%s%inf_%inc_1.00wl_0.50ws_attack.m'%(model, 
                                                          args.mfcc['num_filter'], 
                                                          args.mfcc['num_cep'])
        
        # Eval the model
        runner = Mid_runner_eval(args)
        runner.run()
        print('****************')
        time_end = time.time()
    print('Time comsuming now: %f s'%(time_end-time_start))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Evaluate the peformance of different classifiers")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-cu", "--csv_use", action="store_true", help="Use features and labels from .csv")
    group.add_argument("-pu", "--pkl_use", action="store_true", help="Use features and labels from .pkl")
    
    args = parser.parse_args()
    # For Spyder running. If you use cmd, comment out below line
    
    args.pkl_use = True

    main(args)