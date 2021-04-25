# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 18:38:00 2021
load and check data

@author: hao_y
"""
import pandas as pd
import numpy as np

sensor_placement = ['chest','forearm','head','shin','thigh','upperarm','waist']
actions = ['climbingdown','climbingup','jumping','lying',
           'running','sitting','standing','walking']

filepath = '../proband1/data/acc_'+actions[7]+'_'+sensor_placement[4]+'.csv'
x = pd.read_csv(filepath)

print(x[:5])

import matplotlib.pyplot as plt
def plot_raw(rawdata):
    fig, ax = plt.subplots(3, 1)
    ts = np.arange(0,rawdata.shape[0],step=1)
    ax[0].plot(ts, rawdata[:, 0], 'k-')
    ax[1].plot(ts, rawdata[:, 1], 'r-')
    ax[2].plot(ts, rawdata[:, 2], 'g-')
    plt.show()
    
plot_raw(x.iloc[:,[2,3,4]].values)

# =============================================================================
# basic pre-processing of data
# 1. interpolation
# 2. low-pass filter the noise
# =============================================================================

x_raw = x.iloc[:,[1,2,3,4]].values

x_raw[:,0] = x_raw[:,0] - x_raw[0,0]

fs = 50 #sensor sampling rate
from scipy.interpolate import interp1d
def interpolate_rawdata (data, ts):
    """Make the data evenly sampled by data interpolation
    INPUT: 
        data -- input time series data
        ts -- timestamp
    OUTPUT: 
        interpolated_data -- Nx3 array containing the sensor raw data (no timestamp)
                             1-3 column: x,y,z sensor output data
    """
    # Get interpolation function in terms of timestamp and sensor data
    interpolate_f = interp1d(data[:,0], data[:, [1,2,3]], kind='linear', axis=0)

    # note that this variable only contains the sensor data, no timestamp included
    interpolated_data = interpolate_f(ts)
    data = np.hstack((ts.reshape(-1,1),interpolated_data))
    return data

ts = np.arange(x_raw[:,0].min(),x_raw[:,0].max(),step=1000/fs)
x_inter = interpolate_rawdata(x_raw,ts)

from scipy import signal
def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, np.ravel(data))
    return y

def remove_noise(data, fc_lpf,fs):
    """Remove noise from accelerometer data via low pass filter
    INPUT: 
        data -- input accelerometer data Nx3 (x, y, z axis)
        fc_lpf -- low pass filter cutoff frequency
    OUTPUT: 
        lpf_output -- filtered accelerometer data Nx3 (x, y, z axis)
    """

    # the number of accelerometer readings
    num_data = data[:,0].shape[0]
    # lpf_output is used to store the filtered accelerometer data by low pass filter
    lpf_output = np.zeros((num_data, 3))

    # compute linear acceleration for x axis
    acc_X = data[:,0]
    butterfilter_output= butter_lowpass_filter(acc_X, fc_lpf, fs/2)
    lpf_output[:,0] = butterfilter_output.reshape(1, num_data)

    # compute linear acceleration for y axis
    acc_Y = data[:,1]
    butterfilter_output= butter_lowpass_filter(acc_Y, fc_lpf, fs/2)
    lpf_output[:,1] = butterfilter_output.reshape(1, num_data)

    # compute linear acceleration for z axis
    acc_Z = data[:,2]
    butterfilter_output= butter_lowpass_filter(acc_Z, fc_lpf, fs/2)
    lpf_output[:,2] = butterfilter_output.reshape(1, num_data)

    return lpf_output

fc_lpf = 10
x_data = remove_noise(x_inter[:,1:], fc_lpf, fs)

plot_raw(x_data)

# =============================================================================
# plot one axis or magnitude of the acc signal, save its index by selection
# =============================================================================
from matplotlib.widgets import SpanSelector

def cal_acc(acc):
    m = acc.shape[0]
    res = np.zeros(m)
    for i in range(m):
        res[i] = np.sqrt(acc[i,0]**2+acc[i,1]**2+acc[i,2]**2)
    return res


mag1 = cal_acc(x_data)

###################plot a selectable fig##################################
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(211)

x = np.arange(0,mag1.shape[0],step=1)
y = mag1

ax.plot(x, y, '-')
ax.set_title('Press left mouse button and drag to test')

ax2 = fig.add_subplot(212)
line2, = ax2.plot(x, y, '-')
    
def onselect(xmin, xmax):
    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    indmax = min(len(x) - 1, indmax)

    thisx = x[indmin:indmax]
    thisy = y[indmin:indmax]
    line2.set_data(thisx, thisy)
    ax2.set_xlim(thisx[0], thisx[-1])
    ax2.set_ylim(thisy.min(), thisy.max())
    fig.canvas.draw_idle()

    # save
    np.savetxt("text.txt", np.c_[thisx, thisy])

# set useblit True on gtkagg for enhanced performance
span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))

plt.show()   