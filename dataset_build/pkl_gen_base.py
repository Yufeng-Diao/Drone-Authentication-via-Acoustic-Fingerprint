# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 22:40:38 2022

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

def audio_select(originData_path, dic_choose, dic_aban, \
                 date, distance, drone, name_check):
    # Store audio data and label
    audio = []
    audio_label = []
    for root, dirs, files in os.walk(originData_path):
        for i in range(len(files)):
            name, label = name_check.info_detector(files[i],"drone_No")
            if name_check.check_file_choose(name, dic_choose) \
                and not(name_check.check_file_aban(name, dic_aban)) \
                and files[i].find(date)!=-1 and files[i].find(distance)!=-1 \
                and files[i].find(drone)!=-1:
                    
                audio.append(wave.open(root+'/'+files[i]))
                audio_label.append(label)
                print("Choose %s"%files[i])
            else:
                # print("abandon %s"%files[i])
                pass
    return audio, audio_label

def train_eval_split(audio_data, audio_label):
    train_data = []
    eval_data = []
    
    train_label = []
    eval_label = []
    for data in audio_data:
        seg_point = []
        seg_point.append(int(len(data)*0.2))
        seg_point.append(int(len(data)*0.35))
        seg_point.append(int(len(data)*0.65))
        seg_point.append(int(len(data)*0.8))
        
        train_data.append(data[0:seg_point[0]])
        train_data.append(data[seg_point[1]:seg_point[2]])
        train_data.append(data[seg_point[3]:])
        
        eval_data.append(data[seg_point[0]:seg_point[1]])
        eval_data.append(data[seg_point[2]:seg_point[3]])
        
    for i in audio_label:
        train_label.append(i)
        train_label.append(i)
        train_label.append(i)
        eval_label.append(i)
        eval_label.append(i)
    
    return train_data, train_label, eval_data, eval_label

def dic_quick_check(dic_choose, dic_aban, name_check):
    # Check the format of the dic_choose
    if not(name_check.check_dic(dic_choose)):
        print("The format of the dic_choose is wrong!")
        sys.exit()
    # Check the format of the dic_aban
    if not(name_check.check_dic(dic_aban)):
        print("The format of the dic_aban is wrong!")
        sys.exit()

# Declare the setting of MFCC
mfcc_setting = {}
mfcc_setting['num_filter'] = 201
mfcc_setting['num_cep'] = 201
mfcc_setting['winlen'] = 1
mfcc_setting['winstep'] = 1/2
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
                       '_20220401_', '_20220402_', '_20220403_', '_20220404_', '_20220405_',
                       '_2022noise_']

# dic_choose["date"] = ['_20220304_', '_20220328_']
dic_choose["distance"] = ['_1m_','_5m_']
dic_choose['drone_No'] = ['_d1_','_d2_','_d3_','_d4_','_d5_','_d6_','_d7_','_d8_',
                          '_d9_','_d10_','_d11_','_d12_','_d13_','_d14_','_d15_','_d16_',
                          '_d17_','_d18_','_d19_','_d20_','_d21_','_d22_','_d23_','_d24_','_n_']

# Path to find stored data
originData_path = r'E:\1_Research\3_UAV_2\2_data\2_new_data'
# Path to save csv
pkl_savePath = r'E:\1_Research\3_UAV_2\2_data\pkl_test'


if __name__ == '__main__':
    time_start = time.time()
    # Build the databse for iteration
    pkl_database = pd.DataFrame([])
    name_check = idd.FileNameProcessing(name_set_drone)
    dic_quick_check(dic_choose, dic_aban, name_check)
    
    # Save pkl for each date
    for date in dic_choose["date"]:
        for distance in dic_choose['distance']:
            for drone in dic_choose['drone_No']:
                audio_data = []
                audio, audio_label = audio_select(originData_path, 
                                                  dic_choose, dic_aban, 
                                                  date, distance, drone, 
                                                  name_check)
                
                if not audio:
                    print('No drone available in this condition!')
                    continue
                
                for i in audio:
                    audio_data.append(ap.audio_load(i))
                train_data, train_label, eval_data, eval_label = train_eval_split(audio_data, audio_label)
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
    
    name_output = '_%inf_%inc_%.2fwl_%.2fws_%dlim.pkl'%(mfcc_setting['num_filter'],
                                              mfcc_setting['num_cep'],
                                              mfcc_setting['winlen'],
                                              mfcc_setting['winstep'],
                                              mfcc_setting['highfreq_limit'])
    
    pkl_database.to_pickle(pkl_savePath+'/'+name_output)
    time_end = time.time()
    print("Total running time: %f s"%(time_end-time_start))
