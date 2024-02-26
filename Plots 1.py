#!/usr/bin/env python
# coding: utf-8

"""
Generates plots based on the measurements data obtained from running `LTCAMAnalysis.py` ((requires to run c from 0 to 5).

Parameters:
- dir_path (str): The directory path pointing to where the data files folder is located. 
This path is used to access the measurements data and store the generated plots.

"""

import os, glob

dir_path = os.path.join(...) # example -> os.path.join('/scratch', 'work', 'silvap1', 'LTCAM_article')

import numpy as np
import matplotlib.pyplot as plt, matplotlib.font_manager as fmg
import seaborn as sns
from scipy.signal import savgol_filter, medfilt
import matplotlib.ticker as ticker

plt.rc('font', family='serif')
plt.rcParams['mathtext.fontset'] = 'cm'  # Use the Computer Modern font set for math symbols

dpi = 300
fsd = sorted(glob.glob(os.path.join(dir_path,'data','LTCAM','*mat_3d.npy')))
fs = [f[-24:-11] for f in fsd]

# Set figure size in centimeters
fig_width_cm = 5.5  # Width of the figure in centimeters
fig_height_cm = 5.5  # Height of the figure in centimeters
fig_width_in = fig_width_cm / 2.54  # Convert width to inches
fig_height_in = fig_height_cm / 2.54  # Convert height to inches

pcolor = sns.color_palette('rocket', 6)
clab = ['50 mW', '100 mW', '150 mW', '200 mW', '250 mW', '300 mW']

# # Loading variables
# 
# - t: time of recording (measured by thermal camera)
# - h: length of the helix (calculated using a python algorithm)
# - rang1, rang2: angles of the helix at the boundaries (using minimum bounding rectangle method)
# - el1, el2: angles of helix at the boundaries (using fitting of an ellipse, probably gives better results)
# - tempMAX: maximum temperature measured in each instant of time (measured by the thermal camera)

t, h, rang1, rang2, el1, el2, tempMAX = [], [], [], [], [], [], []
avgh = 0
tmin = 1000
tOFF = np.zeros(6)
c0, c1, c2 = (0, 0, 6)
N = 1000
for c in range(c1, c2):
    t.append(np.load(os.path.join(dir_path, 'data','LTCAM',fs[c] + '_time_3d.npy'))) 
    temp = np.load(os.path.join(dir_path, 'processed','LTCAM', 'Measurements', fs[c] + '_measure.npy'))
    h.append(temp[0])
    rang1.append(temp[1])
    rang2.append(temp[2])
    el1.append(temp[3])
    el2.append(temp[4])
    tempMAX.append(temp[5])
    # Calculate tOFF
    tsm = medfilt(tempMAX[c0], kernel_size=7)
    dT = np.diff(tsm)
    dT2 = np.diff(dT)
    threshold = -0.3
    extr = np.where(dT2 < threshold)[0]
    for l in range(len(extr)):
        if t[c0][extr[l]] > 60:
            tOFF[c0] = extr[l] + 1
            if t[c0][extr[l]] < tmin:
                tmin = t[c0][extr[l]]
            break
    c0 += 1


# # Temperature plots
# 
# In the previous loading variables, 1 point or interest is calculated: the point where temperature drops drastically. After checking the plots it seems the rise of the max temperature is also easy to spot since there is no delay. The reason why these points need to be calculated lies in the method used for comparing the effect on the laser power in the sample XXX. Pedro triggered recording of the thermal camera ON and informs Sioban. When Pedro hears the shutter of the laser (click sound), Pedro triggers the timer. When the timer reaches close the 1 minute (60 seconds) Pedro makes a count down to trigger of the laser. Afterwards, 30 seconds are still recorded. Therefore, laser should be OFF in the measurement, then ON for 60 seconds, and again OFF. So, the plots can be aligned exactly when the laser is turned OFF because there's a significant drop in temperature. I've calculated the extreme points and selected the first extreme drop after the 60 seconds (drops should occur moreless between 60 and 65 seconds, depending on how in sync Pedro and Siohban. Analysing the region where plots should increase temperature, at time = 0s, it looks some points start ascending before 0, but less than 1 second.

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi = 150)
fig.patch.set_facecolor('None')
c0 = 0
on_color = 'green'   
off_color = 'red' 

for c in range(c1, c2):
    x = np.linspace(t[c0].min(), t[c0].max(), N)
    y = np.interp(x, t[c0], tempMAX[c0])
    ysm = savgol_filter(y, window_length=21, polyorder=2, mode='nearest')
    ti = t[c0][int(tOFF[c0])] - 60 #- tmin
    ax.plot(x-ti, ysm, color = pcolor[c0], zorder=2, lw=1, alpha=1,label = clab[c0])
    c0 += 1
    
ax.plot([60, 60], [20, 100], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -1)
ax.plot([0, 0], [20, 100], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -1)
    
ax.legend(loc='best',prop=fmg.FontProperties(size=5))
ax.tick_params(axis='both', labelsize=7)
ax.axvspan(0, 60, facecolor='gray', alpha=0.1, zorder=1)
ax.set_xlim((-5,90))
ax.set_ylim((20, 90))
ax.set_xticks(np.arange(0, 95, 15));
ax.set_yticks(np.arange(20, 91, 10));
ax.set_ylabel('Temperature (°C)', fontsize=7)
ax.set_xlabel('Time (s)', fontsize=7);

plt.savefig(os.path.join(dir_path, 'processed','LTCAM', 'Plots', 'temperature_m734.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0);

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in*2/3), dpi = 150)
c0 = 0
on_color = 'green'   
off_color = 'red' 
tempmax = np.zeros(6)

for c in range(c1, c2):
    x = np.linspace(t[c0].min(), t[c0].max(), N)
    y = np.interp(x, t[c0], tempMAX[c0])
    ysm = savgol_filter(y, window_length=21, polyorder=2, mode='nearest')
    ti = t[c0][int(tOFF[c0])] - 60
    ax.plot(50*(c+1), max(ysm), color = pcolor[c0], zorder=2, marker='.', alpha=1,label = clab[c0])
    tempmax[c0] = max(ysm)
    c0 += 1
    
coeff1 = np.polyfit(50*np.linspace(1,3,3), tempmax[0:3], 1)  # Perform a linear fit (degree=1)
coeff2 = np.polyfit(50*np.linspace(4,6,3), tempmax[3:6], 1)  # Perform a linear fit (degree=1)
xl = np.linspace(0, 350, 100)
ax.plot(xl, coeff1[0] * xl + coeff1[1], 'k--', zorder = -1, lw=0.5)
ax.plot(xl, coeff2[0] * xl + coeff2[1], 'k-.', zorder = -1, lw=0.5)

ax.set_xlim((0,325))
ax.set_xticks(np.arange(0, 301, 50));
ax.set_ylim((20,90))
ax.set_yticks(np.arange(20, 91, 10));
ax.tick_params(axis='both', labelsize=7)
ax.set_ylabel('Max Temperature (°C)', fontsize=7);
ax.set_xlabel('Laser power (mW)', fontsize=7);
plt.savefig(os.path.join(dir_path, 'processed', 'LTCAM', 'Plots', 'maxtemperature_m734.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0)

# # Actuation VS time
# 
# Variable of time t does not change linearly. In order to smooth data we need to create a time variable (x) that changes linearly in time and then interpolate the actuation variable (y):
# - x: goes from t[c0].min() to t[c0].max() by N = 1000
# - y: interpolate the actuation for the x values
# 
# Next, I've used Savgol filtering. Both time variables are subtracted with ti which aligns the experiments at the point the temperature drastically drops (laser is turned off).

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi = 150)
fig.patch.set_facecolor('None')
c0 = 0
L0 = 1 # 1.26 cm
lm = 0
for k in range(c2): lm += h[k][0]
lm /= c2
px = 1332 / 1056 / lm * 10

for c in range(c1, c2):
    f0 = 1 / h[c0][0] * L0
    x = np.linspace(t[c0].min(), t[c0].max(), N)
    y = np.interp(x, t[c0], h[c0])
    y = px * (y - y[0])
    ysm = savgol_filter(y, window_length=21, polyorder=2, mode='nearest')
    ti = t[c0][int(tOFF[c0])] - 60
    ax.plot(x-ti, ysm, color = pcolor[c0], label = clab[c], alpha=1, zorder=-1, linewidth=1)
    c0 += 1
ax.plot([60, 60], [0, 300], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -1)
ax.plot([0, 0], [0, 300], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -5)
    
ax.axvspan(0, 60, facecolor='gray', alpha=0.1, zorder=1)
ax.legend(loc='best',prop=fmg.FontProperties(size=5))
ax.tick_params(axis='both', labelsize=7)
ax.set_ylabel('$\mathrm{\Delta L}$ (mm)', fontsize=7);
ax.set_xlabel('Time (s)', fontsize=7);
ax.set_xlim((-5,90))
ax.set_ylim((-0.25, 4.7))
ax.set_xticks(np.arange(0, 95, 15));
ax.set_yticks(np.arange(0.0, 5, 1));
minor_locator = ticker.FixedLocator(np.arange(0,4.6,0.5))
ax.yaxis.set_minor_locator(minor_locator);

plt.savefig(os.path.join(dir_path, 'processed', 'LTCAM', 'Plots', 'actuation_m734.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0);

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in*2/3), dpi = 150)
c0 = 0
L0 = 1 # 1.26 cm
lm = 0
for k in range(c2): lm += h[k][0]
lm /= c2
px = 1332 / 1056 / lm * 10 # pixel conversion

arcmax = np.zeros(6)
for c in range(c1, c2):
    f0 = 1 / h[c0][0] * L0
    x = np.linspace(t[c0].min(), t[c0].max(), N)
    y = np.interp(x, t[c0], h[c0])
    y = px * (y - y[0])
    ysm = savgol_filter(y, window_length=21, polyorder=2, mode='nearest')
    ti = t[c0][int(tOFF[c0])] - 60
    ax.plot(50*(c+1), max(ysm), color = pcolor[c0], label = clab[c], alpha=1, zorder=1, marker='.')
    arcmax[c0] = max(ysm)
    c0 += 1
ax.tick_params(axis='both', labelsize=7)
ax.set_ylabel('Max $\mathrm{\Delta L}$ (mm)', fontsize=7);
ax.set_xlabel('Laser power (mW)', fontsize=7);
ax.set_xlim((0,325))
ax.set_ylim((0.0, 5.0))
ax.set_xticks(np.arange(0, 325, 50));
ax.set_yticks(np.arange(0.0, 5.1, 1));
minor_locator = ticker.FixedLocator(np.arange(0,4.6,0.5))
ax.yaxis.set_minor_locator(minor_locator)
coeff = np.polyfit(50*np.linspace(1,6,6), arcmax, 1)  # Perform a linear fit (degree=1)
xl = np.linspace(0, 350, 100)
ax.plot(xl, coeff[0] * xl + coeff[1], 'k--', zorder = -1, lw=0.5)

plt.savefig(os.path.join(dir_path, 'processed', 'LTCAM', 'Plots', 'actuationMAX_m734.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0);


# # Angle VS time
# 
# First thing that is needed to do is to load and fix the angles. Ellipses angles change between 0 and 180, so when angle is bellow 0, it will jump to a value below 180 degrees. First thing we need to do is all angles bigger than 90.
# Next we need to interpolate the angles variable for the new time variable. And again a Savgol filter is applied (with a rather small window, so the smoothing factor is not big).
# 
# The variation between angles is represented.
# 
# Some interesting things to notice:
# - 50 mW bends around 10 deg and then unbends when the laser goes off. All other intensities, there is unbending after 10-20 seconds.
# - When the laser is off, if angle is positive, helices don't go immediately to straight configuration but instead make a bending with negative angle. If angle is negative, there's no oscillation returning.
# - For laser intensities 250 and 300 mW, when the angle dips into the negative angle values, helices oscillate.

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi = 150)
fig.patch.set_facecolor('None')
c0 = 0
for c in range(c1, c2):
    ang1 = el1[c0]
    ang2 = el2[c0]
    ang1[ang1>100] -= 180
    ang2[ang2>100] -= 180
    x = np.linspace(t[c0].min(), t[c0].max(), N)
    y1 = np.interp(x, t[c0], ang1)
    y2 = np.interp(x, t[c0], ang2)
    ysm1 = savgol_filter(y1, window_length=21, polyorder=2, mode='nearest')
    ysm2 = savgol_filter(y2, window_length=21, polyorder=2, mode='nearest')
    ti = t[c0][int(tOFF[c0])]-60
    dy = ysm2-ysm1
    dy -= dy[0]
    ax.plot(x-ti, dy, color = pcolor[c0], label = clab[c], alpha=1, zorder=-1, linewidth=1)
    c0 += 1

ax.plot([60, 60], [-100, 300], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -1)
ax.plot([0, 0], [-100, 300], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -2)
    
ax.axvspan(0, 60, facecolor='gray', alpha=0.1, zorder=1)
ax.legend(loc='upper right',prop=fmg.FontProperties(size=5))
ax.tick_params(axis='both', labelsize=7)
ax.set_ylabel(r'$\Delta \theta$ (°)', fontsize=7);
ax.set_xlabel('Time (s)', fontsize=7);
ax.set_xlim((-5,90))
ax.set_xticks(np.arange(0, 95, 15));
ax.set_ylim((-30,115))
ax.set_yticks(np.arange(-30, 105, 30));
minor_locator = ticker.FixedLocator(np.arange(-30,106,15))
ax.yaxis.set_minor_locator(minor_locator)
plt.savefig(os.path.join(dir_path, 'processed', 'LTCAM', 'Plots', 'angle_m734.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0)

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in*2/3), dpi = 150)
fig.patch.set_facecolor('None')
c0 = 0
ymax = np.zeros(6)
for c in range(c1, c2):
    ang1 = el1[c0]
    ang2 = el2[c0]
    ang1[ang1>100] -= 180
    ang2[ang2>100] -= 180
    x = np.linspace(t[c0].min(), t[c0].max(), N)
    y1 = np.interp(x, t[c0], ang1)
    y2 = np.interp(x, t[c0], ang2)
    ysm1 = savgol_filter(y1, window_length=21, polyorder=2, mode='nearest')
    ysm2 = savgol_filter(y2, window_length=21, polyorder=2, mode='nearest')
    ti = t[c0][int(tOFF[c0])]-60
    dy = ysm2-ysm1
    dy -= dy[0]
    ymax[c0] = max(dy) 
    ax.plot(50*(c+1), max(dy), color = pcolor[c0], label = clab[c], alpha=1, zorder=1, marker='.')
    c0 += 1
ax.tick_params(axis='both', labelsize=7)
coeff = np.polyfit(50*np.linspace(1,6,6), ymax, 1)  # Perform a linear fit (degree=1)
xl = np.linspace(0, 350, 100)
ax.plot(xl, coeff[0] * xl + coeff[1], 'k--', zorder = -1, lw=0.5)
ax.set_ylabel(r'Max $\Delta \theta$ (°)', fontsize=7);
ax.set_xlabel('Laser power (mW)', fontsize=7);
ax.set_xlim((0,325))
ax.set_xticks(np.arange(0, 301, 50));
ax.set_ylim((0,115))
ax.set_yticks(np.arange(0, 105, 30));
minor_locator = ticker.FixedLocator(np.arange(-30,106,15))
ax.yaxis.set_minor_locator(minor_locator)
plt.savefig(os.path.join(dir_path, 'processed','LTCAM', 'Plots', 'maxangle_m734.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0)