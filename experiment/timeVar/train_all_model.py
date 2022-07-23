# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:43:18 2022

@author: Eidos
"""
import argparse
import os
import sys
import re
import time 
import yaml
# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if not top_path in sys.path:
    sys.path.append(top_path)

from runners import Mid_runner_train
from toolbox.name_set import drone_set
from toolbox.name_set import name_set_list
from toolbox import train_tool

def main(args, config):
    # The settign of the mfcc
    args.mfcc = config['mfcc_setting']
    # Path to find stored data
    args.originData_path = config['mfcc_setting']
    # Path to store trained model
    args.output_path = config['output_path']
    # args.csv_savePath = r''
    args.pkl_savePath = config['pkl_savePath']
    
    # Predefine the key of the dic
    args.dic_choose = dict([(k,[]) for k in name_set_list])
    args.dic_aban = dict([(k,[]) for k in name_set_list])
    
    args.dic_choose["date"] = ['_20220304_', '_20220307_', '_20220312_', '_20220318_', '_20220319_',
                              '_20220327_', '_20220328_', '_20220329_', '_20220330_', '_20220331_', 
                              '_20220401_', '_20220402_', '_20220403_', '_20220404_', '_20220405_']
    
    # args.dic_choose["drone_No"] = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_','_d7_','_d8_',
    #                               '_d9_','_d10_','_d11_','_d12_','_d13_','_d14_','_d15_','_d16_',
    #                               '_d17_','_d18_','_d19_','_d20_','_d21_','_d22_','_d23_','_d24_']
    args.dic_choose["drone_No"] = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_','_d7_','_d8_']
    
    time_start = time.time()
    # Train all models
    for root, dirs, files in os.walk(args.pkl_savePath):
        for i in range(len(files)):
            
            args.pkl_fileName = files[i]
            args.mfcc['winlen'] = float(re.search('\d+\.?\d*',re.search('_.{1,4}wl_', files[i]).group()).group())
            args.mfcc['winstep'] = float(re.search('\d+\.?\d*',re.search('_.{1,4}ws_', files[i]).group()).group())
            for j in range(8):
                train_tool.mutual_exclusive(args, j)
                args.output_name = train_tool.model_name(args)
        
                runner = Mid_runner_train(args)
                runner.run()
                time_end = time.time()
                print('Time comsuming now: %f s'%(time_end-time_start))


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
    
    with open(os.path.join(top_path, 'config/config_timeVar.yml'),'r') as f:
        content = f.read()
        config = yaml.load(content, Loader=yaml.SafeLoader)
        
    main(args, config)