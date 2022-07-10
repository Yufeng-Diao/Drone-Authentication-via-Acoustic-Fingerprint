# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 18:16:21 2021

@author: Eidos
"""

import wave
import sys
import os

# Add the top level directory in system path
top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not top_path in sys.path:
    sys.path.append(top_path)

import numpy as np
from scipy import signal

from toolbox import pyeidos_fft as e_fft

def audio_load(audio):
    params = audio.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = audio.readframes(nframes)
    waveData = np.frombuffer(strData,dtype=np.int16)
    waveData = np.reshape(waveData,[nframes,nchannels])
    wave_data_left = waveData[:,0]    
    return wave_data_left

def audio_filter(audio_data, cutOff_freq, filter_type='highpass',N=8, sample_rate=44100):
    Wn = 2*cutOff_freq/sample_rate
    b, a = signal.butter(N, Wn, filter_type)
    audio_filt = signal.filtfilt(b, a, audio_data)
    return audio_filt
    
def time2nfft(timestep,fs):
    sa_point = timestep*fs
    nfft = 0
    
    for i in range(100):
        nfft = 2**i
        if (nfft >= sa_point):
            break
    
    return nfft

def time2nfft_win(timestep,fs):
    winlen = timestep
    winstep = timestep/2
    sa_point = timestep*fs
    nfft = 0
    
    for i in range(100):
        nfft = 2**i
        if (nfft >= sa_point):
            break
    
    return winlen, winstep, nfft

def audio_segment(audio_wav, time_step=0.8, time_shift=0.1):
    '''

    Parameters
    ----------
    audio_wav : wav
        single or multi channel
    time_step : float or int, optional
        the time length per sub-audio. The default is 0.8.
    time_shift : float or int
        the time shift per sub-audio.
    
    Returns
    -------
    wave_data_seg : array
        Batch * Data

    '''
    
    audio = [None]
    params = [None]
    nchannels, sampwidth, framerate, nframes = [None], [None], [None], [None]
    strData = [None]
    waveData = [None]
    
    wave_data_left = [None]
    wave_data_seg = []
    
    audio[0] = audio_wav
    
    audio_load(0, audio, params, nchannels, sampwidth, framerate, nframes, strData, waveData, wave_data_left)
    wave_data_left[0] = wave_data_left[0].reshape(-1,1)
    
    seg_length = int(time_step*framerate[0])
    
    seg_move = int(time_shift*framerate[0])
    
    seg_number = int((len(wave_data_left[0])-seg_length)/seg_move)+1
    
    for i in range(seg_number):
        wave_data_seg.append(wave_data_left[0][i*seg_move:i*seg_move+seg_length,0])
    wave_data_seg = np.array(wave_data_seg)
    return wave_data_seg
    
def audio_fft(audio_clip, nfft, origin_normalize=False, fft_normalize=True):
    '''
    
    Parameters
    ----------
    audio_clip : array
        Batch * Data
    nfft : int
        n point fft

    Returns
    -------
    audio_fft : array
        Batch * Data

    '''
    # 对原始数据归一化
    if origin_normalize:
        audio_clip_max = audio_clip.max(1).reshape(-1,1)
        audio_clip_min = audio_clip.min(1).reshape(-1,1)
        audio_clip = (audio_clip-audio_clip_min)/(audio_clip_max-audio_clip_min)
    
    audio_fft = e_fft.fft_group(audio_clip, nfft)
    
    if fft_normalize:
        audio_fft_max = audio_fft.max(1).reshape(-1,1)
        audio_fft_min = audio_fft.min(1).reshape(-1,1)
        audio_fft = (audio_fft-audio_fft_min)/(audio_fft_max-audio_fft_min)
        
    return audio_fft



if __name__=="__main__":
    import os
    import winsound
    import time
    import matplotlib.pyplot as plt
    
    
    fs = 1000
    
    N = 2**15
    omega_range = np.array(range(0,N)).reshape(-1,1)
    
    omega=omega_range*2*np.pi/N;
    
    Omega = omega*fs;
    
    f = Omega/2/np.pi;
    
    y_number = 3
    y = [None]*y_number
    
    x = np.arange(-1,1,0.001)
    y[0] = 5*np.cos(2*np.pi*100*x)+1
    y[1] = 10*np.cos(2*np.pi*200*x)+2
    y[2] = 15*np.cos(2*np.pi*300*x)+3
    
    y_np = np.array(y).reshape(len(y),-1)
    
    y_fft = audio_fft(y_np,N)
    
    for i in range(y_number):
        plt.subplot(y_number,1,i+1)
        plt.plot(f[0:int(N/2)], y_fft[i, 0:int(N/2)])

