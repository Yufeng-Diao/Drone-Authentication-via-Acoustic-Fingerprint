# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 11:35:02 2022

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

def train_eval_split_noise(audio_data, audio_label, snr):
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
        
        data = wgn_add(data, snr)
        
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
        
def wgn(x, snr):
    P_signal = np.sum(abs(x)**2, dtype = np.int64)/len(x)
    P_noise = P_signal/10**(snr/10.0)
    return np.random.randn(len(x)) * np.sqrt(P_noise)

def wgn_add(x, snr):
    return x + wgn(x, snr)#.astype(np.short)