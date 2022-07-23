# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:35:37 2022

@author: Eidos
"""

import argparse
import os
import sys
import time
# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not top_path in sys.path:
    sys.path.append(top_path)
    
import numpy as np
import pandas as pd
import wave
import yaml

import toolbox.info_detector_drone as idd
from toolbox.MFCC_extract import mfcc_extract
from toolbox.name_set import name_set_drone
from toolbox.name_set import name_set_list
from toolbox import audio_processing as ap
from toolbox import pkl_gen_tool as pgt

with open(os.path.join(top_path, 'config/config_filterVar_gen.yml'),'r') as f:
    content = f.read()
    config = yaml.load(content, Loader=yaml.SafeLoader)

# Declare the setting of MFCC
mfcc_setting = config['mfcc_setting']
# Path to find stored data
originData_path = config['originData_path']
# Path to save csv
pkl_savePath = config['pkl_savePath']

dic_choose = dict([(k,[]) for k in name_set_list])
dic_aban = dict([(k,[]) for k in name_set_list])

# Decide which features will be extracted. This version only considers date.
dic_choose["date"] = ['_20220304_', '_20220307_', '_20220312_', '_20220318_', '_20220319_',
                       '_20220327_', '_20220328_', '_20220329_', '_20220330_', '_20220331_', 
                       '_20220401_', '_20220402_', '_20220403_', '_20220404_', '_20220405_']

# dic_choose["date"] = ['_20220304_', '_20220328_']
dic_choose["distance"] = ['_1m_','_5m_']
dic_choose['drone_No'] = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_','_d7_','_d8_',
                          '_d9_','_d10_','_d11_','_d12_','_d13_','_d14_','_d15_','_d16_',
                          '_d17_','_d18_','_d19_','_d20_','_d21_','_d22_','_d23_','_d24_']


if __name__ == '__main__':
    time_start = time.time()
    for _ in range(50):
        mfcc_setting['num_filter'] = mfcc_setting['num_filter'] + 5
        mfcc_setting['num_cep'] = mfcc_setting['num_filter']
        
        # Build the databse for iteration
        pkl_database = pd.DataFrame([])
        name_check = idd.FileNameProcessing(name_set_drone)
        pgt.dic_quick_check(dic_choose, dic_aban, name_check)
        
        # Save csv for each date
        for date in dic_choose["date"]:
            for distance in dic_choose['distance']:
                for drone in dic_choose['drone_No']:
                    audio_data = []
                    audio, audio_label = pgt.audio_select(originData_path, 
                                                      dic_choose, dic_aban, 
                                                      date, distance, drone, 
                                                      name_check)
                    
                    if not audio:
                        print('No drone available in this condition!')
                        continue
                    
                    for i in audio:
                        audio_data.append(ap.audio_load(i))
                    
                    train_data, train_label, eval_data, eval_label = pgt.train_eval_split(audio_data, audio_label)
                    # Generate train data
                    train_pd = pgt.mul_df_gen(train_data, train_label, mfcc_setting, 
                                              date, distance, drone, process = 'train')
                    pkl_database = pkl_database.append(train_pd)
                    # Generate eval data
                    eval_pd = pgt.mul_df_gen(eval_data, eval_label, mfcc_setting, 
                                             date, distance, drone, process = 'eval')
                    pkl_database = pkl_database.append(eval_pd)
        
        name_output = '_%inf_%inc_%.2fwl_%.2fws_%dlim.pkl'%(mfcc_setting['num_filter'],
                                                  mfcc_setting['num_cep'],
                                                  mfcc_setting['winlen'],
                                                  mfcc_setting['winstep'],
                                                  mfcc_setting['highfreq_limit'])
        
        pkl_database.to_pickle(pkl_savePath+'/'+name_output)
        time_end = time.time()
        print("Total running time of building this dataset: %f s"%(time_end-time_start))
