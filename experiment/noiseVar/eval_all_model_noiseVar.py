# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 21:50:27 2022

@author: Eidos
"""

import argparse
import time
import os
import sys
import yaml
# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if not top_path in sys.path:
    sys.path.append(top_path)
    
import numpy as np
import pandas as pd

from runners import Mid_runner_eval
from experiment.timeVar.mid_runner_eval_timeVar import Mid_runner_eval_timeVar
from toolbox.name_set import drone_set
from toolbox.name_set import name_set_csv
from toolbox.name_set import name_set_list

def main(args, config):
    # The settign of the mfcc
    args.mfcc = config['mfcc_setting']
    # Path to find stored data
    # args.originData_path = r'E:\1_Research\3_UAV_2\2_data\2_new_data'
    # Path to store trained model
    args.model_path = config['output_path']
    # Path to save or load the features and labels
    args.csv_savePath = config['csv_savePath']
    # The Path of pkl file
    args.pkl_savePath = config['pkl_savePath']
    
    snr = -8.25
    
    # Predefine the key of the dic
    args.dic_choose = dict([(k,[]) for k in name_set_list])
    args.dic_aban = dict([(k,[]) for k in name_set_list])
    
    args.dic_choose["date"] = ['_20220304_', '_20220307_', '_20220312_', '_20220318_', '_20220319_',
                              '_20220327_', '_20220328_', '_20220329_', '_20220330_', '_20220331_', 
                              '_20220401_', '_20220402_', '_20220403_', '_20220404_', '_20220405_']
    
    args.dic_choose['drone_No'] = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_','_d7_','_d8_',
                              '_d9_','_d10_','_d11_','_d12_','_d13_','_d14_','_d15_','_d16_',
                              '_d17_','_d18_','_d19_','_d20_','_d21_','_d22_','_d23_','_d24_']
    
    # args.dic_choose["drone_No"] = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_','_d7_','_d8_']
    
    model_list = ['_QDA_', '_LDA_', '_LSVM_', '_SVM_', '_KNN_', '_DT_', '_RF_', '_GNB_']
    # Create the header of csv file
    accuracy_list = model_list.copy()
    accuracy_list.insert(0,'')
    accuracy_list = np.array(accuracy_list).reshape(1,-1)
    accuracy_pd = pd.DataFrame(accuracy_list)
    accuracy_pd.to_csv(args.csv_savePath+'/'+'noiseVar.csv', header=False, index=False, mode='a')
    
    
    # accuracy_pd_all = pd.DataFrame(columns =  ['_QDA_', '_LDA_', '_LSVM_', '_SVM_', '_KNN_', '_DT_', '_RF_', '_GNB_'])
    
    time_start = time.time()
    for _ in range(93):
        accuracy_list = []
        snr = snr + 0.25
        args.pkl_fileName ='_%inf_%inc_1.00wl_0.50ws_8000lim_%.2fdB.pkl'%(args.mfcc['num_filter'], 
                                                                   args.mfcc['num_filter'],
                                                                   snr)
        accuracy_list.append(snr)
        try:
            for model in model_list:
                args.model_name = '%s%inf_%inc_1.00wl_0.50ws_.m'%(model, 
                                                                  args.mfcc['num_filter'], 
                                                                  args.mfcc['num_cep'])
                
                # Train the model
                runner = Mid_runner_eval_timeVar(args)
                accuracy = runner.run()
                accuracy_list.append(accuracy)
            # store result
            accuracy_list = np.array(accuracy_list).reshape(1,-1)
            accuracy_pd = pd.DataFrame(accuracy_list)
            
            accuracy_pd.to_csv(args.csv_savePath+'/'+'noiseVar.csv', header=False, index=False, mode='a')
            time_end = time.time()
            print('Time comsuming now: %f s'%(time_end-time_start))
        except FileNotFoundError:
            print('missing file:', args.pkl_fileName)
            continue
    # accuracy_pd_all.to_csv(args.csv_savePath+'/'+'timeVar.csv', header=True, index=True)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Evaluate the peformance of different classifiers")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-cu", "--csv_use", action="store_true", help="Use features and labels from .csv")
    group.add_argument("-pu", "--pkl_use", action="store_true", help="Use features and labels from .pkl")
    
    args = parser.parse_args()
    # For Spyder running. If you use cmd, comment out below line
    
    args.pkl_use = True

    with open(os.path.join(top_path, 'config/4_noiseVar/config_noiseVar.yml'),'r') as f:
        content = f.read()
        config = yaml.load(content, Loader=yaml.SafeLoader)
        
    main(args, config)