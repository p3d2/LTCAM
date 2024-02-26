#!/usr/bin/env python
# coding: utf-8

"""
Generates plots based on the measurements data obtained from running `LTCAMAnalysis.py` (requires to run c from 6 to 13).

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

pcolor = sns.color_palette("colorblind",8)
lcolor = sns.color_palette("pastel",8)

clab = ['ZSØ1B', 'ZSØ1C', 'ZSØ2B', 'ZSØ2C', 'ZZØ2B', 'ZZØ2A', 'ZSØ2A', 'ZSØ1A']

t, h, rang1, rang2, el1, el2, tempMAX = [], [], [], [], [], [], []
avgh = 0
tmin = 1000
tOFF = np.zeros(8)
c0, c1, c2 = (0, 6, 14)
N = 1000
for c in range(c1, c2):
    t.append(np.load(os.path.join(dir_path, 'data','LTCAM',fs[c] + '_time_3d.npy'))) 
    temp = np.load(os.path.join(dir_path, 'processed', 'LTCAM', 'Measurements', fs[c] + '_measure.npy'))
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
        if t[c0][extr[l]] > 30:
            tOFF[c0] = extr[l] + 1
            if t[c0][extr[l]] < tmin:
                tmin = t[c0][extr[l]]
            break
    c0 += 1


# # Temperature VS time plots
# 

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi = 150)
fig.patch.set_facecolor('None')
c0 = [13, 6, 7, 12, 8, 9, 11, 10]
px =  0.08949096568803461 #1332 / 1056 / h[6][0] * 10

pl = sns.color_palette("colorblind", 3)
ls = ['-', '--', ':']
ms = ['o', '^', 's']

for c in range(len(c0)):
    q = int(c0[c]-6)
    x = np.linspace(t[q].min(), t[q].max(), N)
    y = np.interp(x, t[q], tempMAX[q])
    ysm = savgol_filter(y, window_length=41, polyorder=2, mode='interp')
    ti = t[q][int(tOFF[q])] - 30 
    if 'ZS' in clab[q]:
        continue
    ax.plot(t[q]-ti, tempMAX[q], color = pl[int(c/3)], label = clab[q], alpha=1, zorder=-1, linewidth=1, linestyle=ls[np.mod(c,3)], marker=ms[np.mod(c,3)], markersize=0)
    
for k in range(6):
    ax.plot([30*k, 30*k], [20, 120], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -1)
    if np.mod(k,2) == 0:
        ax.axvspan(30*k, 30*(k+1), facecolor='gray', alpha=0.1, zorder=-2)
ax.legend(loc='best',prop=fmg.FontProperties(size=5), bbox_to_anchor=(1.0, 1.0))
ax.tick_params(axis='both', labelsize=7)
ax.set_xlim((-5,185))
ax.set_ylim((20,90));
ax.set_ylabel('Temperature (°C)', fontsize=7)
ax.set_xlabel('Time (s)', fontsize=7);
plt.savefig(os.path.join(dir_path, 'processed', 'LTCAM', 'Plots', 'temperature_ZZ.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0)

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi = 150)
fig.patch.set_facecolor('None')
c0 = [13, 6, 7, 12, 8, 9, 11, 10]
px =  0.08949096568803461 #1332 / 1056 / h[6][0] * 10

pl = sns.color_palette("colorblind", 3)
ls = ['-', '--', ':']
ms = ['o', '^', 's']

for c in range(len(c0)):
    q = int(c0[c]-6)
    x = np.linspace(t[q].min(), t[q].max(), N)
    y = np.interp(x, t[q], tempMAX[q])
    ysm = savgol_filter(y, window_length=41, polyorder=2, mode='interp')
    ti = t[q][int(tOFF[q])] - 30 #- tmin
    if 'ZZ' in clab[q]:
        continue
    ax.plot(t[q]-ti, tempMAX[q], color = pl[int(c/3)], label = clab[q], alpha=1, zorder=-1, linewidth=1, linestyle=ls[np.mod(c,3)], marker=ms[np.mod(c,3)], markersize=0)
    
for k in range(6):
    ax.plot([30*k, 30*k], [20, 120], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -1)
    if np.mod(k,2) == 0:
        ax.axvspan(30*k, 30*(k+1), facecolor='gray', alpha=0.1, zorder=-2)
ax.legend(loc='best',prop=fmg.FontProperties(size=5), bbox_to_anchor=(1.0, 1.0))
ax.tick_params(axis='both', labelsize=7)
ax.set_xlim((-5,180))
ax.set_ylim((20, 90))
ax.set_xticks(np.arange(0, 181, 30));
ax.set_yticks(np.arange(20, 91, 10));
ax.set_ylabel('Temperature (°C)', fontsize=7)
ax.set_xlabel('Time (s)', fontsize=7);
plt.savefig(os.path.join(dir_path, 'processed', 'LTCAM', 'Plots', 'temperature_ZS.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0)


# # Actuation VS time

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi = 150)
fig.patch.set_facecolor('None')
c0 = [13, 6, 7, 12, 8, 9, 11, 10]
px =  0.08949096568803461 #1332 / 1056 / h[6][0] * 10

pl = sns.color_palette("colorblind", 3)
ls = ['-', '--', ':']
ms = ['o', '^', 's']

for c in range(len(c0)):
    q = int(c0[c]-6)
    f0 = 1 / h[q][0]
    x = np.linspace(t[q].min(), t[q].max(), N)
    y = np.interp(x, t[q], h[q])
    y = px * (y - y[0])
    ysm = savgol_filter(y, window_length=21, polyorder=2, mode='interp')
    ti = t[q][int(tOFF[q])] - 30
    if 'ZS' in clab[q]:
        continue
    ax.plot(x-ti, ysm, color = pl[int(c/3)], label = clab[q], alpha=1, zorder=-1, linewidth=1, linestyle=ls[np.mod(c,3)], marker=ms[np.mod(c,3)], markersize=0)

for k in range(6):
    ax.plot([30*k, 30*k], [-20, 120], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -1)
    if np.mod(k,2) == 0:
        ax.axvspan(30*k, 30*(k+1), facecolor='gray', alpha=0.1, zorder=-2)
ax.legend(loc='best',prop=fmg.FontProperties(size=5), bbox_to_anchor=(1.0, 1.0))
ax.set_ylabel('$\mathrm{\Delta L}$ (mm)', fontsize=7)
ax.tick_params(axis='both', labelsize=7)
ax.set_xlabel('Time (s)', fontsize=7)
ax.set_xlim((-5,180))
ax.set_ylim((-15,0.5))
ax.set_xticks(np.arange(0, 181, 30))
ax.set_yticks(np.arange(-14, 0.1, 2));
minor_locator = ticker.FixedLocator(np.arange(-15,5,0.5))
ax.yaxis.set_minor_locator(minor_locator)
plt.savefig(os.path.join(dir_path, 'processed', 'LTCAM', 'Plots', 'actuation_ZZ.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0)

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi = 150)
fig.patch.set_facecolor('None')
c0 = [13, 6, 7, 12, 8, 9, 11, 10]
px =  0.08949096568803461 #1332 / 1056 / h[6][0] * 10

pl = sns.color_palette("colorblind", 3)
ls = ['-', '--', ':']
ms = ['o', '^', 's']

tw = [5.2, 5.2, 5.2, 6.0, 5.6, 4.5]

for c in range(len(c0)):
    q = int(c0[c]-6)
    f0 = 1 / h[q][0]
    x = np.linspace(t[q].min(), t[q].max(), N)
    y = np.interp(x, t[q], h[q])
    y = px * (y - y[0])
    ysm = savgol_filter(y, window_length=21, polyorder=2, mode='interp')
    ti = t[q][int(tOFF[q])] - 30
    if 'ZZ' in clab[q]:
        continue
    ax.plot(x-ti,6*ysm/tw[c], color = pl[int(c/3)], label = clab[q], alpha=1, zorder=-1, linewidth=1, linestyle=ls[np.mod(c,3)], marker=ms[np.mod(c,3)], markersize=0)

for k in range(6):
    ax.plot([30*k, 30*k], [-20, 120], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -1)
    if np.mod(k,2) == 0:
        ax.axvspan(30*k, 30*(k+1), facecolor='gray', alpha=0.1, zorder=-2)
ax.legend(loc='best',prop=fmg.FontProperties(size=5), bbox_to_anchor=(1.0, 1.0))
ax.tick_params(axis='both', labelsize=7)
ax.set_ylabel('$\mathrm{\Delta L}$ (mm)', fontsize=7)
ax.set_xlabel('Time (s)', fontsize=7)
ax.set_xlim((-5,180))
ax.set_ylim((-0.25,4.7))
ax.set_xticks(np.arange(0, 181, 30))
ax.set_yticks(np.arange(0, 4.7, 1));
minor_locator = ticker.FixedLocator(np.arange(-15,5,0.5))
ax.yaxis.set_minor_locator(minor_locator)
plt.savefig(os.path.join(dir_path, 'processed', 'LTCAM', 'Plots', 'actuation_ZS.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0)

# # Angle VS time

plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi = 150)
fig.patch.set_facecolor('None')
pl = sns.color_palette("colorblind", 3)
ls = ['-', '--', ':']
ms = ['o', '^', 's']
c0 = [13, 6, 7, 12, 8, 9, 11, 10]

for c in range(len(c0)):
    q = int(c0[c]-6)
    ang1 = el1[q]
    ang2 = el2[q]
    ang1[ang1>100] -= 180
    ang2[ang2>100] -= 180
    x = np.linspace(t[q].min(), t[q].max(), N)
    y1 = np.interp(x, t[q], ang1)
    y2 = np.interp(x, t[q], ang2)
    ysm1 = savgol_filter(y1, window_length=21, polyorder=2, mode='interp')
    ysm2 = savgol_filter(y2, window_length=21, polyorder=2, mode='interp')
    ti = t[q][int(tOFF[q])]-30
    dy = ysm2-ysm1
    dy -= dy[0]
    if 'ZZ' in clab[q]:
        continue
    ax.plot(x-ti, dy, color = pl[int(c/3)], label = clab[q], alpha=1, zorder=-1, linewidth=1, linestyle=ls[np.mod(c,3)], marker=ms[np.mod(c,3)], markersize=0)

for k in range(6):
    ax.plot([30*k, 30*k], [-180, 180], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -1)
    if np.mod(k,2) == 0:
        ax.axvspan(30*k, 30*(k+1), facecolor='gray', alpha=0.1, zorder=-2)
ax.legend(loc='best',prop=fmg.FontProperties(size=5), bbox_to_anchor=(1.0, 1.0))
ax.set_ylabel(r'$\Delta \theta$ (°)', fontsize=7)
ax.tick_params(axis='both', labelsize=7)
ax.set_xlabel('Time (s)', fontsize=7)
ax.set_xlim((-5,180))
ax.set_ylim((-30,115))
ax.set_xticks(np.arange(0, 181, 30))
ax.set_yticks(np.arange(-30, 110, 30));
minor_locator = ticker.FixedLocator(np.arange(-30,106,15))
ax.yaxis.set_minor_locator(minor_locator)
plt.savefig(os.path.join(dir_path, 'processed', 'LTCAM', 'Plots', 'angle_ZS.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0)


plt.close()
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi = 150)
fig.patch.set_facecolor('None')
pl = sns.color_palette("colorblind", 3)
ls = ['-', '--', ':']
ms = ['o', '^', 's']
c0 = [13, 6, 7, 12, 8, 9, 11, 10]

for c in range(len(c0)):
    q = int(c0[c]-6)
    ang1 = el1[q]
    ang2 = el2[q]
    ang1[ang1>100] -= 180
    ang2[ang2>100] -= 180
    x = np.linspace(t[q].min(), t[q].max(), N)
    y1 = np.interp(x, t[q], ang1)
    y2 = np.interp(x, t[q], ang2)
    ysm1 = savgol_filter(y1, window_length=21, polyorder=2, mode='interp')
    ysm2 = savgol_filter(y2, window_length=21, polyorder=2, mode='interp')
    ti = t[q][int(tOFF[q])]-30
    dy = ysm2-ysm1
    dy -= dy[0]
    if 'ZS' in clab[q]:
        continue
    ax.plot(x-ti, dy, color = pl[int(c/3)], label = clab[q], alpha=1, zorder=-1, linewidth=1, linestyle=ls[np.mod(c,3)], marker=ms[np.mod(c,3)], markersize=0)

for k in range(6):
    ax.plot([30*k, 30*k], [-180, 180], color='black', linewidth=0.25, linestyle='dashed', dashes=(10, 10), zorder = -1)
    if np.mod(k,2) == 0:
        ax.axvspan(30*k, 30*(k+1), facecolor='gray', alpha=0.1, zorder=-2)
ax.legend(loc='best',prop=fmg.FontProperties(size=5), bbox_to_anchor=(1.0, 1.0))
ax.tick_params(axis='both', labelsize=7)
ax.set_ylabel(r'$\Delta \theta$ (°)', fontsize=7)
ax.set_xlabel('Time (s)', fontsize=7)
ax.set_xlim((-5,180))
ax.set_ylim((-30,30))
ax.set_xticks(np.arange(0, 181, 30))
ax.set_yticks(np.arange(-30, 31, 30));
minor_locator = ticker.FixedLocator(np.arange(-30,106,15))
ax.yaxis.set_minor_locator(minor_locator)
plt.savefig(os.path.join(dir_path, 'processed', 'LTCAM', 'Plots', 'angle_ZZ.pdf'), format='pdf', dpi=dpi, bbox_inches='tight',pad_inches=0)