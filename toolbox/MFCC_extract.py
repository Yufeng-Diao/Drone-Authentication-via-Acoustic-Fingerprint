# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:08:35 2021

@author: Eidos
"""

import numpy as np
import os
import sys
# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not top_path in sys.path:
    sys.path.append(top_path)
    
from toolbox.mfcc_base import mfcc
from toolbox.mfcc_base import delta

import toolbox.audio_processing as ap
from sklearn.preprocessing import normalize

def mfcc_extract(audio_data, audio_label, num_filter=101, num_cep=101, 
                 winlen=0.8, winstep=0.8, fs=44100,
                 mfcc_d1_switch=True, mfcc_d2_switch=True, first_feat = False, 
                 feature_norm = True, highfreq = 8000):

    # intrinsic parameters
    wave_feature_origin = None
    wave_feature_all = None
    wave_label_all = None
    # audio_data = []
    
    # Calculate the nfft point number according to window length
    nfft = ap.time2nfft(winlen,fs)
    # print(nfft)
    
    # for i in range(0,len(audio)):
    #     audio_data.append(ap.audio_load(audio[i]))
    
    # 0th dimension MFCC
    print('Start extracting mfcc features ...')
    wave_feature_all, wave_label_all = mfcc_zero_dimension(audio_data, audio_label, 
                                                           num_filter, num_cep,
                                                           winlen, winstep,
                                                           fs, nfft, highfreq)
    print('Finish extracting mfcc features')
    # For extraction high dimensional features
    wave_feature_origin = wave_feature_all
    # Delete first features
    if not first_feat:
        wave_feature_all = np.delete(wave_feature_all, 0, axis=1)
    if feature_norm:
        wave_feature_all = normalize(wave_feature_all, axis=1, norm='l2')
        
    # 1st dimension MFCC
    print('Start extracting d1 features ...')
    if mfcc_d1_switch:
        mfcc_hd_1 = delta(wave_feature_origin, 1)
        # Delete first features
        if not first_feat:
            mfcc_hd_1 = np.delete(mfcc_hd_1, 0, axis=1)
        if feature_norm:
            mfcc_hd_1 = normalize(mfcc_hd_1, axis=1, norm='l2')
        wave_feature_all = np.hstack((wave_feature_all, mfcc_hd_1))
    print('Finish extracting d1 features')
    
    # 2nd dimension MFCC
    print('Start extracting d2 features ...')
    if mfcc_d2_switch:
        mfcc_hd_2 = delta(wave_feature_origin, 2)
        # Delete first features
        if not first_feat:
            mfcc_hd_2 = np.delete(mfcc_hd_2, 0, axis=1)
        if feature_norm:
            mfcc_hd_2 = normalize(mfcc_hd_2, axis=1, norm='l2')
        wave_feature_all = np.hstack((wave_feature_all, mfcc_hd_2))
    print('Finish extracting d2 features')
    
    return wave_feature_all, wave_label_all

def mfcc_zero_dimension(audio_data, audio_label, num_filter=101, num_cep=101, 
                        winlen=0.8, winstep=0.8, fs=44100, nfft = 512, highfreq = 8000):
    
    wave_feature_all = None
    wave_label_all = None
    
    for i in range(0,len(audio_data)):
        print('Processing file No. ', i)
        # filter low frequency noise
        # wave_data_left[i] = ap.audio_filter(wave_data_left[i], cutOff_freq=200)
        # extract MFCC
        wave_feature = mfcc(audio_data[i], samplerate=fs, numcep=num_cep, winlen=winlen, winstep=winstep,
                                   nfilt=num_filter, nfft=nfft, lowfreq=0, highfreq=highfreq, preemph=0.97, 
                                   winfunc=np.hamming)
        # delete 0th MFCC, which is related to the energy (DC component)
        # wave_feature = np.delete(wave_feature[i],0,axis=1)
        # calculate the label
        wave_label = np.zeros(len(wave_feature)).reshape(-1,1)+audio_label[i]
        # initiate the iteration
        if wave_feature_all is None:
            wave_feature_all = wave_feature
            wave_label_all = wave_label
        else:
            wave_feature_all = np.vstack((wave_feature_all, wave_feature))
            wave_label_all = np.vstack((wave_label_all, wave_label))
            
    return wave_feature_all, wave_label_all

def mfcc_high_dimension(wave_feature_origin, dimension = 1):
    # initiate the iteration
    mfcc_hd = delta(wave_feature_origin, dimension)
    
    return mfcc_hd

# def peak_norm():
    

if __name__ == "__main__":
    import wave
    audio = []
    audio_label = []
    audio.append(wave.open(r"E:\1_Research\3_UAV_2\1_playground\1_mfcc\_UAV_20210806_d1_hover_1m_100%_1_.WAV"))
    audio_label.append(1)
    
    wave_feature_all, wave_label_all = mfcc_extract(audio, audio_label, num_filter=101, num_cep=101, 
                                                    winlen=0.8, winstep=0.8, fs = 44100,
                                                    mfcc_d1_switch=False, mfcc_d2_switch=False)

    
