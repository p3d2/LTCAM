#!/usr/bin/env python
# coding: utf-8

import os, glob, math

c = 1 # select file to analyse
dir_path = os.path.join('/scratch', 'work', 'silvap1', 'article_rods')
#dir_path = os.path.join(r'\\data.triton.aalto.fi', 'work', 'silvap1', 'article_rods')

fsd_vid = sorted(glob.glob(os.path.join(dir_path,'data','Rotating','*mp4')))
fs = [f[-29:-7] for f in fsd_vid]

fsd_ir = sorted(glob.glob(os.path.join(dir_path,'data','Rotating','*npz')))
fs_ir = [f[-17:-4] for f in fsd_vid]

import cv2
from skimage.morphology import skeletonize
import numpy as np
import matplotlib.pyplot as plt, matplotlib.font_manager as fmg
from matplotlib import colormaps
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
import networkx as nx

dpi = 300

if not os.path.exists(os.path.join(dir_path,'temp')):
    os.makedirs(os.path.join(dir_path,'temp'))
if not os.path.exists(os.path.join(dir_path,'processed')):    
    os.makedirs(os.path.join(dir_path,'processed'))
if not os.path.exists(os.path.join(dir_path,'processed','Rotating')):  
    os.makedirs(os.path.join(dir_path,'processed','Rotating'))
    os.makedirs(os.path.join(dir_path,'processed','Rotating','Measurements'))
    os.makedirs(os.path.join(dir_path,'processed','Rotating','Plots'))
    os.makedirs(os.path.join(dir_path,'processed','Rotating','Videos'))

def clean_directory(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path} due to {e}")

def intersection_area(contour, region):
    x1, y1, x2, y2 = region
    
    # Create an image filled with zeros, and an image filled with ones in the shape of the contour
    mask = cv2.fillPoly(np.zeros((y2-y1, x2-x1)), [contour - [x1, y1]], 255).astype(np.uint8)
    
    # Calculate area by counting non-zero pixels
    area = cv2.countNonZero(mask)
    return area

def select_contours(contours, region):

    # Calculate intersection area for each contour and pair with contour
    contours_with_area = [(contour, intersection_area(contour, region)) for contour in contours]
    
    # Filter and sort by intersection area
    selected_contours = [contour for contour, area in contours_with_area if area > 0]
    selected_contours.sort(key=lambda contour: intersection_area(contour, region), reverse=True)
    
    return selected_contours

def calculate_angle(c1, c2, x, y):
    dy = y - c2
    dx = x - c1
    return math.atan2(dy, dx)

def residuals(params, points):
    c1, c2, r = params
    x, y = points.T
    return np.sqrt((x - c1)**2 + (y - c2)**2) - r

def fit_circle(points):
    x, y = points.T
    params_initial = (np.mean(x), np.mean(y), np.mean(np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2)))
    res = least_squares(residuals, params_initial, args=(points,))
    c1, c2, r = res.x
    return c1, c2, r

def get_adjacent_nodes(node, shape):
    adjacent_nodes = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            r, c = node[0] + i, node[1] + j
            if r >= 0 and r < shape[0] and c >= 0 and c < shape[1]:
                adjacent_nodes.append((r, c))
    return adjacent_nodes

def plot_scn():
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    im = ax.imshow(mat, cmap=cmap, vmin = 20, vmax = 90)
    cmap.set_under('white') # SET color below vmin
    cmap.set_over('white')  # SET color over vmax
    cb = fig.colorbar(im, ax=ax, pad=0.0, aspect=10, shrink=0.67)
    ax.axis('off')  
    font = fmg.FontProperties(size=14)
    for text in cb.ax.get_yticklabels():
        text.set_font_properties(font)
        text.set_color(fg_color)
    cb.outline.set_edgecolor(fg_color)
    cb.ax.yaxis.set_tick_params(color=fg_color)

def save_img(k):
    plot_scn()
    plt.savefig(os.path.join(dir_path, 'temp', str(k+1).zfill(3)+'.png'), dpi=dpi, bbox_inches='tight',pad_inches=0)
    plt.close('all')

clean_directory(os.path.join(dir_path, 'temp'))


# Calculate angular speed of rotating platform
video = cv2.VideoCapture(fsd_vid[c])

if c < 3:
    y_start, y_end = 300, 900  
    x_start, x_end = 700, 1300  
else:
    y_start, y_end = 300, 900 
    x_start, x_end = 850, 1450  
    
fps = 6
# Loop through all frames in the video

TOT = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
vx = np.zeros(int(np.ceil(TOT/fps)))
vy = np.zeros(int(np.ceil(TOT/fps)))
time_f = np.zeros(int(np.ceil(TOT/fps)))
for k in range(TOT):
    ret, frame = video.read()
    if not ret:
        break  # Break the loop if we are at the end of the video

    # Process every [fps] frames, i.e., every second
    if k % fps == 0:
        
        time_f[int(k/fps)] = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        frame_c = frame[y_start:y_end, x_start:x_end]
        c_frame = cv2.cvtColor(frame_c, cv2.COLOR_BGR2LAB)
        
        # Calculate angular speed
        l_channel = cv2.cvtColor(frame_c, cv2.COLOR_BGR2HSV)[:,:,1]
        _, th = cv2.threshold(l_channel, 50, 255, cv2.THRESH_BINARY)
        
        cv2.circle(th, (290, 300), 110, (0, 0, 0), thickness=-1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dil1 = cv2.dilate(th, kernel, iterations = 1)
        th = cv2.erode(dil1, kernel, iterations = 1)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        vx[int(k/fps)] = int(moments['m10'] / moments['m00'])
        vy[int(k/fps)] = int(moments['m01'] / moments['m00'])

# Circumnutation tracking

# Get the frames per second (fps) of the video
vx2 = np.zeros(int(np.ceil(TOT/fps)))
vy2 = np.zeros(int(np.ceil(TOT/fps)))
vx2b0 = np.zeros(int(np.ceil(TOT/fps)))
vy2b0 = np.zeros(int(np.ceil(TOT/fps)))
vx2b1 = np.zeros(int(np.ceil(TOT/fps)))
vy2b1 = np.zeros(int(np.ceil(TOT/fps)))

vx3 = np.zeros(int(np.ceil(TOT/fps)))
vy3 = np.zeros(int(np.ceil(TOT/fps)))
time2 = np.zeros(int(np.ceil(TOT/fps)))
npoints = np.zeros(int(np.ceil(TOT/fps)))
inters = []
fsize = 40

video = cv2.VideoCapture(fsd_vid[c])
for k in range(TOT):
    ret, frame = video.read()
    if not ret:
        break  # Break the loop if we are at the end of the video

    # Process every [fps] frames, i.e., every second
    if k % fps == 0:
        time2[int(k/fps)] = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        frame_c = frame[y_start:y_end, x_start:x_end]
        l_channel = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
        
        
        _, th = cv2.threshold(l_channel, 50, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        er1 = cv2.erode(th, kernel, iterations = 2)
        dil1 = cv2.dilate(er1, kernel, iterations = 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
        dil2 = cv2.dilate(dil1, kernel, iterations = 1)
        er2 = cv2.erode(dil2, kernel, iterations = 1)
        th = er2
        
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200 and not cv2.pointPolygonTest(cnt, (0, 0), False) == 1]
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        
        # Skeletonization
        binary_mask = np.zeros_like(frame_c, dtype=np.uint8) # Create an empty image to fill the contour
        cv2.drawContours(binary_mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED) # Fill the contour to create a binary mask
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY) # Convert to grayscale and binary
        _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        skeleton = skeletonize(binary_mask // 255, method='lee').astype(np.uint8) #img_as_ubyte(skeleton, method = 'lee')
        
        # Create a graph from the skeleton
        graph = nx.Graph()
        for index in np.ndindex(skeleton.shape):
            if skeleton[index]:
                graph.add_node(index) 
        for node in graph.nodes:
            for neighbor in get_adjacent_nodes(node, skeleton.shape):
                if skeleton[neighbor]:
                    graph.add_edge(node, neighbor)
        (vx2b0[int(k/fps)], vy2b0[int(k/fps)]), (vx2b1[int(k/fps)], vy2b1[int(k/fps)]) = list(graph.nodes)[0], list(graph.nodes)[-1]
        
        # Fit ellipse
        
        ellipse = cv2.fitEllipse(largest_contour)
        scaled = tuple(x * 0.5 for x in ellipse[1])
        ellipse = (ellipse[0], scaled) + ellipse[2:]
        ellipse_img = np.zeros_like(skeleton)
        cv2.ellipse(ellipse_img, ellipse, 255, -1)  # -1 fills the ellipse
        intersection = cv2.bitwise_and(ellipse_img, skeleton)
        
        # Linear regression
        y_sk, x_sk = np.where(intersection > 0)
        npoints[int(k/fps)] = len(y_sk)
        xreg = x_sk.reshape(-1, 1)
        yreg = y_sk.reshape(-1, 1)
        sk = np.array([(xi, yi) for xi, yi in zip(x_sk, y_sk)]).reshape((-1,1,2))#
        model = LinearRegression().fit(xreg, yreg)
        m = model.coef_[0]
        b = model.intercept_
        
        minx = np.min(largest_contour[:,:,0])
        maxx = np.max(largest_contour[:,:,0])
        intersections = []
        for q1 in range(minx, maxx + 1):
            q2 = m[0] * q1 + b[0]
            if cv2.pointPolygonTest(largest_contour, (q1, q2), False) >= 0:
                intersections.append((q1, q2))
        
        if len(intersections) > 2:
            min_point = min(intersections, key=lambda t: t[0])  # Select point with min x
            max_point = max(intersections, key=lambda t: t[0])  # Select point with max x
            true_intersections = [min_point, max_point]
        else:
            true_intersections = intersections
            print(intersections, "needs debugging")
        inters.append(true_intersections)
        # Calculate line endpoints
        x1p = 0
        y1p = int(b.item())
        x2p = intersection.shape[1]
        y2p = int(m.item() * x2p + b.item())
        
        moments = cv2.moments(largest_contour)
        vx2[int(k/fps)] = int(moments['m10'] / moments['m00'])
        vy2[int(k/fps)] = int(moments['m01'] / moments['m00'])
        cv2.ellipse(frame_c, ellipse, (0,220,20), 1)
        cv2.line(frame_c, (x1p, y1p), (x2p, y2p), (255,255,255), 1)
        cv2.polylines(frame_c, sk, True, (60, 200, 20), 2)
    
        _, th = cv2.threshold(l_channel, 17, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        er1 = cv2.erode(th, kernel, iterations = 2)
        dil1 = cv2.dilate(er1, kernel, iterations = 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
        dil2 = cv2.dilate(dil1, kernel, iterations = 1)
        er2 = cv2.erode(dil2, kernel, iterations = 1)
        th = dil1
        
    
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            xc, yc, wc, hc = cv2.boundingRect(largest_contour)
        
        if len(contours)>0:
            moments = cv2.moments(contours[0])
            vx3[int(k/fps)] = int(moments['m10'] / moments['m00'])
            vy3[int(k/fps)] = int(moments['m01'] / moments['m00'])
        
            
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    
        cv2.drawContours(frame_c, contours, -1, (60, 20, 220), 1)
        # Convert frames to RGB for visualization with matplotlib
        frame_rgb = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        ax.plot(vy2b1[int(k/fps)], vx2b1[int(k/fps)], 'rx')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(dir_path, 'temp', str(k+1).zfill(5)+'.png'), dpi=dpi, bbox_inches='tight',pad_inches=0)
        plt.close('all')

video_name = os.path.join(dir_path,'processed','Rotating','Videos', 'video_' + fs[c] + '.avi')
images = sorted([img for img in  os.listdir(os.path.join(dir_path, 'temp')) if img.endswith(".png")])
frame = cv2.imread(os.path.join(dir_path, 'temp', images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 10.0, (width,height))

for l in range(len(images)):
    video.write(cv2.imread(os.path.join(dir_path, 'temp', images[l])))
      
cv2.destroyAllWindows()
video.release()

np.savez(os.path.join(dir_path, 'processed', 'Rotating', 'Measurements', fs[c] + '_measure.npz'), vx=vx, vy=vy, time=time_f, vx2=vx2, vy2=vy2, vx3=vx3, vy3=vy3, 
         vx4a = vx2b0, vy4a = vy2b0, vx4b = vx2b1, vy4b = vy2b1, time2=time2, inters=inters, npoints=npoints)

# IR video read
clean_directory(os.path.join(dir_path, 'temp'))

load_file = np.load(fsd_ir[c])
matload = load_file['vid_mat']
t = load_file['t']
fg_color = 'black'
cmap = colormaps.get_cmap('turbo').copy()
dpi = 300

t0, c0 = 0, 0
med = np.median(np.diff(t))
t_max = np.zeros(len(matload))
for k in range(len(matload)):
    mat = matload[k]
    t_max[k] = np.amax(mat)
    if int(t[k]) > t0:
        dvid = int(t[k]-t0)
        t0 = int(t[k])
        if dvid > 1: print('Repeated', dvid)
        for l in range(dvid):
            save_img(c0)
            c0 += 1

np.savez(os.path.join(dir_path, 'processed', 'Rotating', 'Measurements', fs[c] + '_measureIR.npz'), t_max=t_max, time=t)
            
video_name = os.path.join(dir_path,'processed','Rotating', 'Videos', 'video_ir_' + fs_ir[c] + '.avi')
images = sorted([img for img in os.listdir(os.path.join(dir_path, 'temp')) if img.endswith(".png")])
frame = cv2.imread(os.path.join(dir_path, 'temp', images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 10.0, (width,height))

for l in range(c0):
    video.write(cv2.imread(os.path.join(dir_path, 'temp', images[l])))

cv2.destroyAllWindows()
video.release()