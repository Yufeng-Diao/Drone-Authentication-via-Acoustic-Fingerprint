# -*- coding: utf-8 -*-
"""
Created on Tue May  3 22:03:06 2022

@author: Eidos
"""

import argparse
import os
import sys
# import re
import time 
import yaml
# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if not top_path in sys.path:
    sys.path.append(top_path)

import numpy as np

from experiment.attack.mid_runner_train_attack import Mid_runner_train
# from toolbox.name_set import drone_set
from toolbox.name_set import name_set_drone
from toolbox.name_set import name_set_list
from toolbox import train_tool

def main(args, config):
    # The settign of the mfcc
    args.mfcc = config['mfcc_setting']
    # Path to find stored data
    args.originData_path = config['originData_path']
    # Path to store trained model
    args.output_path = config['output_path']
    args.csv_savePath = config['csv_savePath']
    args.pkl_savePath = config['pkl_savePath']
    args.pkl_fileName = config['pkl_fileName']
    
    # Predefine the key of the dic
    args.dic_choose = dict([(k,[]) for k in name_set_list])
    args.dic_aban = dict([(k,[]) for k in name_set_list])
    
    args.dic_choose["date"] = ['_20220304_', '_20220307_', '_20220312_', '_20220318_', '_20220319_',
                              '_20220327_', '_20220328_', '_20220329_', '_20220330_', '_20220331_', 
                              '_20220401_', '_20220402_', '_20220403_', '_20220404_', '_20220405_',
                              '_2022noise_']
    
    dic_reg, dic_attack, args.bg_type = dic_gen()
    dic_string(dic_reg, 'dic_reg')
    dic_string(dic_attack, 'dic_attack')
    dic_string(args.bg_type, 'args.bg_type')
    
    args.dic_choose["drone_No"] = dic_reg + args.bg_type
    args.dic_choose["drone_No"].sort(key=name_set_drone["drone_No"].index)
    # print(args.dic_choose["drone_No"])

    time_start = time.time()
    # Iterate on all methods
    for j in range(8):
        train_tool.mutual_exclusive(args, j)
        args.output_name = train_tool.model_name(args, 'attack')

        runner = Mid_runner_train(args)
        runner.run()
        time_end = time.time()
        print('Time comsuming now: %f s'%(time_end-time_start))


def dic_gen():
    dic_all = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_','_d7_','_d8_',
                '_d9_','_d10_','_d11_','_d12_','_d13_','_d14_','_d15_','_d16_',
                '_d17_','_d18_','_d19_','_d20_','_d21_','_d22_','_d23_','_d24_']
    # Build the dic of register drone
    dic_reg = list(np.random.choice(dic_all, 8, replace=False))
    dic_reg.sort(key=dic_all.index)
    dic_reg.append('_n_')
    # The dic of rest drone
    dic_16 = list(set(dic_all).difference(set(dic_reg)))
    dic_16.sort(key=dic_all.index)
    # The dic of attack drone
    dic_attack = list(np.random.choice(dic_16, 8, replace=False))
    dic_attack.sort(key=dic_all.index)
    # The dic of background drone
    dic_bg = list(set(dic_16).difference(set(dic_attack)))
    dic_bg.sort(key=dic_all.index)
    
    print('dic_reg = ', dic_reg)
    print('dic_attack = ', dic_attack)
    print('dic_bg = ', dic_bg)
    
    return dic_reg, dic_attack, dic_bg
    
def dic_string(dic, name):
    string = ''
    for drone in dic:
        string = string + drone.replace('_','').replace('d','') + ','
    print('%s is '%(name), string)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train diffrent classifiers")
    # mutually exclusive options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--qda", action="store_true", help='Using QDA')
    group.add_argument("--lda", action="store_true", help='Using LDA')
    group.add_argument("--lsvm", action="store_true", help='Using LSVM')
    group.add_argument("--svm", action="store_true", help='Using SVM')
    group.add_argument("--knn", action="store_true", help='Using KNN')
    group.add_argument("--dt", action="store_true", help='Using DT')
    group.add_argument("--rf", action="store_true", help='Using RF')
    group.add_argument("--gnb", action="store_true", help='Using GNB')
    # Use csv files
    group_2 = parser.add_mutually_exclusive_group()
    group_2.add_argument("-cu", "--csv_use", action="store_true", help="Use features and labels from .csv")
    group_2.add_argument("-pu", "--pkl_use", action="store_true", help="Use features and labels from .pkl")
    # Save model
    parser.add_argument("-ms", "--model_save", action="store_true", help="Save the model")
    
    args = parser.parse_args()
    
    args.qda = True
    args.model_save = True
    args.pkl_use = True
    
    with open(os.path.join(top_path, 'config/5_attack/config_attack.yml'),'r') as f:
        content = f.read()
        config = yaml.load(content, Loader=yaml.SafeLoader)
    
    main(args, config)