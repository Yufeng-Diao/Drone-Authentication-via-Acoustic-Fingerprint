# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 21:48:56 2022

@author: Eidos
"""

import argparse
import numpy as np
import os
import time
import sys
import yaml
# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if not top_path in sys.path:
    sys.path.append(top_path)
    
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
    args.originData_path = config['originData_path']
    # Path to store trained model
    args.model_path = config['output_path']
    # Path to save or load the features and labels
    args.csv_savePath = config['csv_savePath']
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
    
    # args.pkl_fileName = r'_50nf_50nc_0.8wl_0.4ws_8000lim.pkl'
    
    model_list = ['_QDA_', '_LDA_', '_LSVM_', '_SVM_', '_KNN_', '_DT_', '_RF_', '_GNB_']
    
    # accuracy_pd_all = pd.DataFrame(columns =  ['_QDA_', '_LDA_', '_LSVM_', '_SVM_', '_KNN_', '_DT_', '_RF_', '_GNB_'])

    args.mfcc['winlen'] = -0.03
    time_start = time.time()
    for _ in range(1):
        accuracy_list = []
        args.mfcc['winlen'] = args.mfcc['winlen'] + 0.05
        args.mfcc['winstep'] = round(args.mfcc['winlen']/2,2)
        args.pkl_fileName ='_50nf_50nc_%.2fwl_%.2fws_8000lim.pkl'%(args.mfcc['winlen'], 
                                                                   args.mfcc['winstep'])
        for model in model_list:
            args.model_name = '%s50nf_50nc_%.2fwl_%.2fws_.m'%(model, 
                                                              args.mfcc['winlen'], 
                                                              args.mfcc['winstep'])
            
            # Train the model
            runner = Mid_runner_eval_timeVar(args)
            accuracy = runner.run()
            accuracy_list.append(accuracy)
        # store result
        # Generate the new dataframe and transpose it. The shape should be like (1,6)
        accuracy_pd = pd.DataFrame({str(args.mfcc['winlen']):accuracy_list}).T
        # Change the name of the columns. 
        accuracy_pd.columns = model_list
        # Add this new row to the summary table
        # accuracy_pd_all = pd.concat([accuracy_pd_all,accuracy_pd])
        accuracy_pd.to_csv(args.csv_savePath+'/'+'timeVar.csv', header=False, index=False, mode='a')
        time_end = time.time()
        print('Time comsuming now: %f s'%(time_end-time_start))
        
    # accuracy_pd_all.to_csv(args.csv_savePath+'/'+'timeVar.csv', header=True, index=True)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Evaluate the peformance of different classifiers")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-cu", "--csv_use", action="store_true", help="Use features and labels from .csv")
    group.add_argument("-pu", "--pkl_use", action="store_true", help="Use features and labels from .pkl")
    
    args = parser.parse_args()
    # For Spyder running. If you use cmd, comment out below line
    
    args.pkl_use = True
    
    with open(os.path.join(top_path, 'config/config_timeVar.yml'),'r') as f:
        content = f.read()
        config = yaml.load(content, Loader=yaml.SafeLoader)
    
    main(args, config)