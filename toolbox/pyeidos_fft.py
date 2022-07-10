# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 20:55:42 2021

@author: Eidos
"""

import numpy as np

def choose_windows(name='Hamming', N=20):
    # Rect/Hanning/Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Rect':
        window = np.ones(N)
    return window


def fft_single(data, nfft):
    
    data_np = np.array(data).reshape(-1,1)
    
    window = choose_windows(name='Hamming', N=len(data_np)).reshape(-1,1)
    data_np = data_np * window
    
    data_fft = abs(np.fft.fft(data_np,nfft,axis=0)).reshape(-1,1)
    
    dc_number = int(nfft/len(data_np))
    data_fft[0:dc_number] = data_fft[0:dc_number]/2
    
    data_fft = data_fft/(len(data_np)/2)
    
    data_fft = data_fft*2
    return data_fft


def fft_group(data, nfft):
    '''
    
    Parameters
    ----------
    data : array, Batch * Data
        DESCRIPTION.
    nfft : int
        DESCRIPTION.

    Returns
    -------
    data_fft : array, Batch * Data
        DESCRIPTION.

    '''
    
    data_length = data.shape[-1]
    window = choose_windows(name='Hamming', N=data_length).reshape(1,-1)
    data = data.reshape(-1,data_length) * window
    
    data_fft = abs(np.fft.rfft(data,nfft,axis=-1))
    
    dc_number = int(nfft/data_length)
    data_fft[:, 0:dc_number] = data_fft[:, 0:dc_number]/2
    
    data_fft = data_fft/(data_length/2)
    
    data_fft = data_fft*2
    return data_fft

if __name__ == "__main__":
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

    y_fft = fft_group(y_np,N)
    
    for i in range(y_number):
        plt.subplot(y_number,1,i+1)
        plt.plot(f[0:int(N/2)], y_fft[i, 0:int(N/2)])
    
    plt.figure()
    y_fft_single = fft_group(y[0],N)
    plt.plot(f[0:int(N/2)], y_fft_single[0, 0:int(N/2)])




