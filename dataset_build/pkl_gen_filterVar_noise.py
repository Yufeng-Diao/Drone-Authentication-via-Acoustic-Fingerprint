# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:45:33 2022

@author: Eidos
"""

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

import toolbox.info_detector_drone as idd
from toolbox.MFCC_extract import mfcc_extract
from toolbox.name_set import name_set_drone
from toolbox.name_set import drone_set
from toolbox import audio_processing as ap
from toolbox import pkl_gen_tool as pgt

# Declare the setting of MFCC
mfcc_setting = {}
mfcc_setting['num_filter'] = 26
mfcc_setting['num_cep'] = 26
mfcc_setting['winlen'] = 1
mfcc_setting['winstep'] = 0.5
mfcc_setting['fs'] = 44100
mfcc_setting['mfcc_d1_switch'] = True
mfcc_setting['mfcc_d2_switch'] = True
mfcc_setting['highfreq_limit'] = 8000
# All valid key
name_set_list = ['prefix','date','drone_No','state','distance','index','suffix']

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

# Path to find stored data
originData_path = r'E:\1_Research\3_UAV_2\2_data\2_new_data'
# Path to save csv
pkl_savePath = r'E:\1_Research\3_UAV_2\2_data\8_pkl_filterVar_noise'
# Signal to noise ratio
snr = 0


if __name__ == '__main__':
    mfcc_setting['num_filter'] = 21
    mfcc_setting['num_cep'] = 21
    time_start = time.time()
    for _ in range(3):
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
                    train_data, train_label, eval_data, eval_label = pgt.train_eval_split_noise(audio_data, audio_label, snr)
                    # Generate train data
                    train_mfcc, _ = \
                        mfcc_extract(train_data, train_label, 
                                      num_filter = mfcc_setting['num_filter'],
                                      num_cep = mfcc_setting['num_cep'], 
                                      winlen = mfcc_setting['winlen'], 
                                      winstep = mfcc_setting['winstep'], 
                                      fs = mfcc_setting['fs'],
                                      mfcc_d1_switch = mfcc_setting['mfcc_d1_switch'], 
                                      mfcc_d2_switch = mfcc_setting['mfcc_d2_switch'],
                                      first_feat = True, feature_norm = True, 
                                      highfreq = mfcc_setting['highfreq_limit'])
                        
                    train_pd = pd.DataFrame(train_mfcc)
                    # Set the multiIndex of rows
                    train_pd = train_pd.set_index([pd.Series(['train']*len(train_pd)),
                                                   pd.Series([date]*len(train_pd)),
                                                   pd.Series([distance]*len(train_pd)),
                                                   pd.Series([drone]*len(train_pd))])
                    # Set the multiIndex of columns
                    train_pd.columns = pd.MultiIndex.from_product([pd.Series(['0_mfcc','1_mfcc','2_mfcc']),
                                                                   range(mfcc_setting['num_cep'])],
                                                                  names=['dimension','SN'])
                    pkl_database = pkl_database.append(train_pd)
                    
                    eval_mfcc, _ = \
                        mfcc_extract(eval_data, eval_label, 
                                      num_filter = mfcc_setting['num_filter'],
                                      num_cep = mfcc_setting['num_cep'], 
                                      winlen = mfcc_setting['winlen'], 
                                      winstep = mfcc_setting['winstep'], 
                                      fs = mfcc_setting['fs'],
                                      mfcc_d1_switch = mfcc_setting['mfcc_d1_switch'], 
                                      mfcc_d2_switch = mfcc_setting['mfcc_d2_switch'],
                                      first_feat = True, feature_norm = True, 
                                      highfreq = mfcc_setting['highfreq_limit'])
                    
                    eval_pd = pd.DataFrame(eval_mfcc)
                    # Set the multiIndex of rows
                    eval_pd = eval_pd.set_index([pd.Series(['eval']*len(eval_pd)),
                                                 pd.Series([date]*len(eval_pd)),
                                                 pd.Series([distance]*len(eval_pd)),
                                                 pd.Series([drone]*len(eval_pd))])
                    eval_pd.index.names = ['mode', 'date', 'distance', 'drone_No']
                    # Set the multiIndex of columns
                    eval_pd.columns = pd.MultiIndex.from_product([pd.Series(['0_mfcc','1_mfcc','2_mfcc']),
                                                                  range(mfcc_setting['num_cep'])],
                                                                  names=['dimension','SN'])
                    
                    pkl_database = pkl_database.append(eval_pd)
        
        name_output = '_%inf_%inc_%.2fwl_%.2fws_%dlim_%idB.pkl'%(mfcc_setting['num_filter'],
                                                  mfcc_setting['num_cep'],
                                                  mfcc_setting['winlen'],
                                                  mfcc_setting['winstep'],
                                                  mfcc_setting['highfreq_limit'],
                                                  snr)
        
        pkl_database.to_pickle(pkl_savePath+'/'+name_output)
        time_end = time.time()
        print("Total running time of building this dataset: %f s"%(time_end-time_start))
