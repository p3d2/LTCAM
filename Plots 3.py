#!/usr/bin/env python
# coding: utf-8

import os, glob, math

dir_path = os.path.join('/scratch', 'work', 'silvap1', 'article_rods')

#dir_path = os.path.join(r'\\data.triton.aalto.fi', 'work', 'silvap1', 'article_rods')

fsd = sorted(glob.glob(os.path.join(dir_path,'data','Rotating','*npz')))
fsd2 = sorted(glob.glob(os.path.join(dir_path,'processed','Rotating','Measurements', '*npz')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from scipy.optimize import least_squares
import scipy.ndimage

dpi = 300
# Set figure size in centimeters
fig_width_cm = 8 # Width of the figure in centimeters
fig_height_cm = 8  # Height of the figure in centimeters
fig_width_in = fig_width_cm / 2.54  # Convert width to inches
fig_height_in = fig_height_cm / 2.54  # Convert height to inches

def calculate_angle(c1, c2, x, y):
    dy = y - c2
    dx = x - c1
    return math.atan2(dy, dx)

def residuals(params, points):
    c1, c2, r = params
    x, y = points.T
    return np.sqrt((x - c1)**2 + (y - c2)**2) - r

def fit_circle(points, threshold=None, exclude_outliers=True):
    x, y = points.T
    params_initial = (np.mean(x), np.mean(y), np.mean(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2)))
    res = least_squares(residuals, params_initial, args=(points,))
    c1, c2, r = res.x
    
    # Check if exclusion of outliers is requested
    if exclude_outliers and threshold is not None:
        # Calculate residuals for all points
        all_residuals = residuals(res.x, points)
        # Create a mask for points with residuals under the threshold
        mask = np.abs(all_residuals) < threshold
        # Refit the circle using only the inliers
        if np.any(mask):
            res = least_squares(residuals, params_initial, args=(points[mask],))
            c1, c2, r = res.x
    
    return c1, c2, r

def draw_all(load_data, data2, method, v_type, a, v, sigma3):
    # Access the variables from the loaded_data dictionary
    vx = load_data['vx']
    vy = load_data['vy']
    time = load_data['time']
    vx2 = load_data['vx2']
    vy2 = load_data['vy2']
    vx3 = load_data['vx3']
    vy3 = load_data['vy3']
    time2 = load_data['time2']
    inters = load_data['inters']
    npoints = load_data['npoints']
    tt = data2['time']
    t_max = data2['v_max']
    points = np.array([(xi, yi) for xi, yi in zip(vx, vy)])
    c1, c2, r = fit_circle(points, threshold=20.0)
    
    angles_radians = [calculate_angle(c1, c2, x, y) for x, y in points]
    angles_degrees = [math.degrees(theta) for theta in angles_radians]
    
    df_angle = np.diff(angles_degrees)
    df_angle[abs(df_angle > 100)] -= 360  
    ms = np.median(df_angle)/np.median(np.diff(time))
    print(np.median(np.diff(time)),np.median(df_angle),np.median(df_angle)/np.median(np.diff(time)))
    
    # Intersections Method 3
    if method == '2':
        valx = vx3
        valy = vy3
        p0x = valx[0]
        p0y = valy[0]
        valx_sm = scipy.ndimage.gaussian_filter1d(valx - valx[0], sigma=2)
        valy_sm = scipy.ndimage.gaussian_filter1d(valy - valy[0], sigma=2)
        
    if method == '3':
        valx = np.zeros(len(inters))
        valy = np.zeros(len(inters))
        p0x = inters[0][1][0]
        p0y = inters[0][1][1]
            
        for k in range(len(inters)):
            if k > 0:
                diff_v1 = abs(inters[k][v][a] - inters[k-1][v][a])
                diff_v2 = abs(inters[k][1-v][a] - inters[k-1][v][a])
                if diff_v1 > 5 and diff_v2 < diff_v1:
                    v = 1-v
            if v_type == '1-v':
                valx[k] = inters[k][1-v][0] 
                valy[k] = inters[k][1-v][1]
            if v_type == 'v':
                valx[k] = inters[k][1][0] 
                valy[k] = inters[k][1][1]
        valx_sm = scipy.ndimage.gaussian_filter1d(valx - p0x, sigma=2)
        valy_sm = scipy.ndimage.gaussian_filter1d(valy - p0y, sigma=2)    
    
    if method == '4':
        valx = np.zeros(len(inters))
        valy = np.zeros(len(inters))
        for k in range(len(inters)):
            if inters[k][0][1] < inters[k][1][1]:
                v = 0
            else:
                v = 1
            valx[k] = inters[k][1-v][0]
            valy[k] = inters[k][1-v][1]
            p0x = inters[0][1-v][0]
            p0y = inters[0][1-v][1]
            valx_sm = scipy.ndimage.gaussian_filter1d(valx - p0x, sigma=2)
            valy_sm = scipy.ndimage.gaussian_filter1d(valy - p0y, sigma=2)
    
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi = 150)
    ax.scatter(valx_sm, valy_sm, marker='o', c=time[:len(valx_sm)], cmap='turbo', edgecolors='none', alpha=0.5, zorder=1, s=10)
    ax.plot(0, 0,'ko')
    ax.tick_params(axis='both', labelsize=7)
    ax.set_xlim((-200,200))
    ax.set_ylim((200,-200));
    ax.set_ylabel('x', fontsize=7)
    ax.set_xlabel('y', fontsize=7);

    points = np.array([(xi, yi) for xi, yi in zip(valx, valy)])
    angles_radians = [calculate_angle(c1, c2, x, y) for x, y in points]
    angles_degrees3 = [math.degrees(theta) for theta in angles_radians]
    points_sm = np.array([(xi, yi) for xi, yi in zip(valx_sm, valy_sm)])
    angles_radians = [calculate_angle(0, 0, x, y) for x, y in points_sm]
    angles_degrees_sm = [math.degrees(theta) for theta in angles_radians]
    
    
    fig, axs = plt.subplots(3, 1, figsize=(6, 12))  # 3 rows, 1 column     
    ws = ((ms*time+180) % 360 - 180)
    
    axs[0].plot(tt, t_max, c='red', marker='.', ms=1)
    axs[0].set_title('Temperature')
    
    axs[1].plot(time[:len(ws)], ws, 'k.', ms=1)
    tcrop = time[:len(angles_degrees3)]
    axs[1].set_title('Angle')
    
    v_norm = np.zeros(len(valx))
    v_norm[v_norm<1.0] = 0
    for k in range(len(valx)):
        v_norm[k] = np.linalg.norm(np.array((valx_sm[k],valy_sm[k])))
    norm_sm = scipy.ndimage.gaussian_filter1d(v_norm, sigma=sigma3)
    
    crad = 12
    tmask = tcrop[norm_sm>crad]
    angmask = np.array(angles_degrees_sm)[norm_sm>crad]
    angle_diff = np.abs(np.diff(angmask))
    mask = angle_diff > 90
    tmask = np.insert(tmask, np.where(mask)[0] + 1, np.nan)
    angmask = np.insert(angmask, np.where(mask)[0] + 1, np.nan)
    
    axs[1].plot(tmask, angmask, c='darkorange', ls = '-', lw=1)
    
    px = 40.42/600
    axs[2].set_title('Radius')
    axs[2].plot(time[:len(norm_sm)], norm_sm*px)
    
    return time, tt, t_max, angles_degrees_sm, norm_sm, valx_sm, valy_sm

# File 0

q = 0
t1_0, t2_0, temp_0, ang_0, rad_0, x_0, y_0 = draw_all(np.load(fsd[q]), np.load(fsd2[q]), method = '3', v_type='v', a=0, v=1, sigma3=1)

# File 1

q = 1
t1_1, t2_1, temp_1, ang_1, rad_1, x_1, y_1 = draw_all(np.load(fsd[q]), np.load(fsd2[q]), method = '3', v_type='1-v', a=1, v=0, sigma3=4)

# File 2

q = 2
t1_2, t2_2, temp_2, ang_2, rad_2, x_2, y_2 = draw_all(np.load(fsd[q]), np.load(fsd2[q]), method = '3', v_type='v', a=0, v=1, sigma3=1)

# File 3

q = 3
t1_3, t2_3, temp_3, ang_3, rad_3, x_3, y_3 = draw_all(np.load(fsd[q]), np.load(fsd2[q]), method = '4', v_type='v', a=0, v=1, sigma3=5)

# File 4

q = 4
t1_4, t2_4, temp_4, ang_4, rad_4, x_4, y_4 = draw_all(np.load(fsd[q]), np.load(fsd2[q]), method = '4', v_type='v', a=1, v=1, sigma3=1)

# File 5

q = 5
t1_5, t2_5, temp_5, ang_5, rad_5, x_5, y_5 = draw_all(np.load(fsd[q]), np.load(fsd2[q]), method = '4', v_type='v', a=0, v=1, sigma3=5)


# Draw all plots
fig_width_cm = 12 # Width of the figure in centimeters
fig_height_cm = 8  # Height of the figure in centimeters
fig_width_in = fig_width_cm / 2.54  # Convert width to inches
fig_height_in = fig_height_cm / 2.54  # Convert height to inches
    
fig, ax= plt.subplots(2, 3, figsize=(fig_width_in, fig_height_in), dpi = 150, sharex='col', sharey='row')

for k1 in range(2):
    for k2 in range(3):
        ax[k1,k2].axis('square')
        ax[k1,k2].plot(0, 0,'ko',zorder=-1)
        ax[k1,k2].tick_params(axis='both', labelsize=7)
        ax[k1,k2].set_xlim((-200,200))
        ax[k1,k2].set_ylim((200,-200));
        ax[k1,k2].set_xticks(np.arange(-150, 151, 150))
        ax[k1,k2].set_yticks(np.arange(-150, 151, 150))
        ax[k1,k2].hlines(np.arange(-150, 151, 50), -200, 200, colors='gray', linestyles='dashed',zorder=-1,lw=0.5,alpha=0.5)
        ax[k1,k2].vlines(np.arange(-150, 151, 50), -200, 200, colors='gray', linestyles='dashed',zorder=-1,lw=0.5,alpha=0.5)        
        ax[k1,k2].hlines(np.arange(0, 1, 50), -200, 200, colors='black', linestyles='-',zorder=-1,lw=0.5,alpha=0.5)
        ax[k1,k2].vlines(np.arange(0, 1, 50), -200, 200, colors='black', linestyles='-',zorder=-1,lw=0.5,alpha=0.5)        
        ax[k1,k2].set_xticks([])
        ax[k1,k2].set_yticks([])
        
bbox_props = dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.6)            

colrm = 'viridis'
jump = 2
ax[0,0].scatter(x_1[::jump], y_1[::jump], marker='o', c=t1_1[:len(x_1):jump], cmap=colrm, edgecolors='none', alpha=0.5, zorder=1, s=10)
arc = Arc(xy = (0,0), width = 300, height = 300, angle = 0, theta1=-90-3.3, theta2=-90+3.3, color='black', lw=1, capstyle='round', zorder=3)
ax[0,0].add_patch(arc)
ax[0,0].text(0, -175, r'$\omega = {}^\circ/s$'.format(-6.6), ha='center', va='center', fontsize=7, bbox=bbox_props)


ax[0,1].scatter(x_2[::jump], y_2[::jump], marker='o', c=t1_2[:len(x_2):jump], cmap=colrm, edgecolors='none', alpha=0.5, zorder=1, s=10)
arc = Arc(xy = (0,0), width = 300, height = 300, angle = 0, theta1=-90-7.65, theta2=-90+7.65, color='black', lw=1, capstyle='round', zorder=3)
ax[0,1].add_patch(arc)
ax[0,1].text(0, -175, r'$\omega = {}^\circ/s$'.format(-15.3), ha='center', va='center', fontsize=7, bbox=bbox_props)

ax[0,2].scatter(x_0[::jump], y_0[::jump], marker='o', c=t1_0[:len(x_0):jump], cmap=colrm, edgecolors='none', alpha=0.5, zorder=1, s=10)
arc = Arc(xy = (0,0), width = 300, height = 300, angle = 0, theta1=-90-14, theta2=-90+14, color='black', lw=1, capstyle='round', zorder=3)
ax[0,2].add_patch(arc)
ax[0,2].text(0, -175, r'$\omega = {}^\circ/s$'.format(-28.0), ha='center', va='center', fontsize=7, bbox=bbox_props)

ax[1,0].scatter(x_3[::jump], y_3[::jump], marker='o', c=t1_3[:len(x_3):jump], cmap=colrm, edgecolors='none', alpha=0.5, zorder=1, s=10)
arc = Arc(xy = (0,0), width = 300, height = 300, angle = 0, theta1=-90-2.6, theta2=-90+2.6, color='black', lw=1, capstyle='round', zorder=3)
ax[1,0].add_patch(arc)
ax[1,0].text(0, -175, r'$\omega = {}^\circ/s$'.format(5.2), ha='center', va='center', fontsize=7, bbox=bbox_props)

ax[1,1].scatter(x_5[::jump], y_5[::jump], marker='o', c=t1_5[:len(x_5):jump], cmap=colrm, edgecolors='none', alpha=0.5, zorder=1, s=10)
arc = Arc(xy = (0,0), width = 300, height = 300, angle = 0, theta1=-90-6.4, theta2=-90+6.4, color='black', lw=1, capstyle='round', zorder=3)
ax[1,1].add_patch(arc)
ax[1,1].text(0, -175, r'$\omega = {}^\circ/s$'.format(12.8), ha='center', va='center', fontsize=7, bbox=bbox_props)

ax[1,2].scatter(x_4[::jump], y_4[::jump], marker='o', c=t1_4[:len(x_4):jump], cmap=colrm, edgecolors='none', alpha=0.5, zorder=1, s=10)
arc = Arc(xy = (0,0), width = 300, height = 300, angle = 0, theta1=-90-27.3/2, theta2=-90+27.3/2, color='black', lw=1, capstyle='round', zorder=3)
ax[1,2].add_patch(arc)
ax[1,2].text(0, -175, r'$\omega = {}^\circ/s$'.format(27.3), ha='center', va='center', fontsize=7, bbox=bbox_props)

plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Adjust wspace and hspace as needed
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(dir_path, 'processed', 'Rotating', 'Plots', 'circum.pdf'), format='pdf', dpi=dpi)

# Time vs angle

fig_width_cm = 6 # Width of the figure in centimeters
fig_height_cm = 6  # Height of the figure in centimeters
fig_width_in = fig_width_cm / 2.54  # Convert width to inches
fig_height_in = fig_height_cm / 2.54  # Convert height to inches
fig, ax= plt.subplots(6, figsize=(fig_width_in, fig_height_in), dpi = 150, sharex='row')

for k in range(6):
    ax[k].hlines(np.arange(-90, 91, 90), 0, 400, colors='black', linestyles='-',zorder=-1,lw=0.5,alpha=0.5)
    ax[k].set_xlim((0,140))
    ax[k].set_ylim((-180,180))
    ax[k].set_yticks([])
    ax[k].tick_params(axis='both', labelsize=7)
    if k < 5:
        ax[k].set_xticks([])

crad = 4
tmask = t1_1[rad_1>crad]
angmask = np.array(ang_1)[rad_1>crad]
angle_diff = np.abs(np.diff(angmask))
mask = angle_diff > 90
tmask = np.insert(tmask, np.where(mask)[0] + 1, np.nan)
angmask = np.insert(angmask, np.where(mask)[0] + 1, np.nan)
ax[0].scatter(tmask, -angmask, c=tmask, cmap=colrm, ls = '-', s=1, rasterized='True')

tmask = t1_2[rad_2>crad]
angmask = np.array(ang_2)[rad_2>crad]
angle_diff = np.abs(np.diff(angmask))
mask = angle_diff > 90
tmask = np.insert(tmask, np.where(mask)[0] + 1, np.nan)
angmask = np.insert(angmask, np.where(mask)[0] + 1, np.nan)
ax[1].scatter(tmask, -angmask, c=tmask, cmap=colrm, ls = '-', s=1, rasterized='True')

tmask = t1_0[rad_0>crad]
angmask = np.array(ang_0)[rad_0>crad]
angle_diff = np.abs(np.diff(angmask))
mask = angle_diff > 90
tmask = np.insert(tmask, np.where(mask)[0] + 1, np.nan)
angmask = np.insert(angmask, np.where(mask)[0] + 1, np.nan)
ax[2].scatter(tmask, -angmask, c=tmask, cmap=colrm, ls = '-', s=1, rasterized='True')

tmask = t1_3[rad_3>crad]
angmask = np.array(ang_3)[rad_3>crad]
angle_diff = np.abs(np.diff(angmask))
mask = angle_diff > 90
tmask = np.insert(tmask, np.where(mask)[0] + 1, np.nan)
angmask = np.insert(angmask, np.where(mask)[0] + 1, np.nan)
ax[3].scatter(tmask, -angmask, c=tmask, cmap=colrm, ls = '-', s=1, rasterized='True')

t5_c = t1_5[:len(ang_5)]
tmask = t5_c[rad_5>crad]
angmask = np.array(ang_5)[rad_5>crad]
angle_diff = np.abs(np.diff(angmask))
mask = angle_diff > 90
tmask = np.insert(tmask, np.where(mask)[0] + 1, np.nan)
angmask = np.insert(angmask, np.where(mask)[0] + 1, np.nan)
ax[4].scatter(tmask, -angmask, c=tmask, cmap=colrm, ls = '-', s=1, rasterized='True')

tmask = t1_4[rad_4>crad]
angmask = np.array(ang_4)[rad_4>crad]
angle_diff = np.abs(np.diff(angmask))
mask = angle_diff > 90
tmask = np.insert(tmask, np.where(mask)[0] + 1, np.nan)
angmask = np.insert(angmask, np.where(mask)[0] + 1, np.nan)
ax[5].scatter(tmask, -angmask, c=tmask, cmap=colrm, ls = '-', s=1, rasterized='True')
ax[5].set_xlabel('Time (s)', fontsize=7)
plt.savefig(os.path.join(dir_path, 'processed', 'Rotating', 'Plots', 'angles.pdf'), format='pdf', dpi=dpi)