# RUN ALL

import os, glob
dir_path = os.path.join('/scratch', 'work', 'silvap1', 'article_rods')
#dir_path = os.path.join(r'\\data.triton.aalto.fi', 'work', 'silvap1', 'article_rods')
c = 1

import cv2
import numpy as np
import matplotlib.pyplot as plt, matplotlib.font_manager as fmg
from matplotlib import colormaps

fsd_vid = sorted(glob.glob(os.path.join(dir_path,'data','Butterfly','*.mp4')))
fs = [f[-29:-7] for f in fsd_vid]

fsd_ir = sorted(glob.glob(os.path.join(dir_path,'data','Butterfly','*.npz')))
fs_ir = [f[-17:-4] for f in fsd_ir]

if not os.path.exists(os.path.join(dir_path,'temp')):
    os.makedirs(os.path.join(dir_path,'temp'))
if not os.path.exists(os.path.join(dir_path,'processed')):    
    os.makedirs(os.path.join(dir_path,'processed'))
if not os.path.exists(os.path.join(dir_path,'processed','Butterfly')):  
    os.makedirs(os.path.join(dir_path,'processed','Butterfly'))
    os.makedirs(os.path.join(dir_path,'processed','Butterfly','Measurements'))
    os.makedirs(os.path.join(dir_path,'processed','Butterfly','Plots'))
    os.makedirs(os.path.join(dir_path,'processed','Butterfly','Videos'))

dpi = 300
# Set figure size in centimeters
fig_width_cm = 5.5 # Width of the figure in centimeters
fig_height_cm = 5.5  # Height of the figure in centimeters
fig_width_in = fig_width_cm / 2.54  # Convert width to inches
fig_height_in = fig_height_cm / 2.54  # Convert height to inches

def clean_directory(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path} due to {e}")
            
def draw_plots(vx, vy, time, time2, t_max):
    move_n = np.zeros(len(vx))
    for k in range(len(vx)):
        move_n[k] = np.linalg.norm(np.array((vx[k],vy[k]))-np.array((vx[0],vy[0])))
    
    px = 33.4/400
    fig, ax = plt.subplots(2,1,figsize=(fig_width_in, fig_height_in), dpi = 150, sharex = True)
    ax[0].plot(time, move_n * px, 'k', lw = 0.5)
    ax[0].set_xlim((0,230))
    ax[0].set_ylim((0,7.0));
    ax[0].set_ylabel('Displacement (mm)', fontsize=7, labelpad=12)
        
    ax[1].plot(time2, t_max, c='firebrick', lw = 0.25, zorder=-1)
    ax[1].set_xlim((0,230))
    ax[1].set_ylim((20,104));
    ax[1].set_xlabel('Time (s)', fontsize=7)
    ax[1].set_ylabel('Temperature (°C)', fontsize=7);
    for k in range(2):
        ax[k].tick_params(axis='both', labelsize=7)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Adjust wspace and hspace as needed
    plt.tight_layout(pad=0.1)

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
    
dpi = 150
# clear temp folder
clean_directory(os.path.join(dir_path, 'temp'))

# Load the video
video = cv2.VideoCapture(fsd_vid[c])

if c == 0:
    x_start, x_end = 300, 700  
    y_start, y_end = 550, 950
    xc, yc = (200,80)
if c == 1:
    x_start, x_end = 300, 700  
    y_start, y_end = 850, 1250
    xc, yc = (220,300)

# Get the frames per second (fps) of the video
fps = 6

TOT = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
vx = np.zeros(int(TOT/fps))
vy = np.zeros(int(TOT/fps))
time_f = np.zeros(int(TOT/fps))

fsize = 60
for k in range(TOT):
    ret, frame = video.read()
    if not ret:
        break  # Break the loop if we are at the end of the video

    # Process every [fps] frames, i.e., every second
    if k % fps == 0:
        
        time_f[int(k/fps)] = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        frame_c = frame[y_start:y_end, x_start:x_end]
        
        l_channel = cv2.cvtColor(frame_c, cv2.COLOR_BGR2HSV)[:,:,2]
        if k == 0:
            template = l_channel[yc:yc+fsize, xc:xc+fsize]
        res = cv2.matchTemplate(l_channel, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        matched_roi = (max_loc[0], max_loc[1], fsize, fsize)
        vx[int(k/fps)] = max_loc[0] + fsize/2
        vy[int(k/fps)] = max_loc[1] + fsize/2
        cv2.rectangle(frame_c, (matched_roi[0], matched_roi[1]), (matched_roi[0]+fsize, matched_roi[1]+fsize), (0, 255, 0), 1)

        if k == 0:
            fig, ax = plt.subplots(dpi=300)
            ax.imshow(cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB))
            ax.plot(vx[int(k/fps)], vy[int(k/fps)], 'x')
            ax.set_xlim((0,400))
            ax.set_ylim((0,400))
        
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
        # Convert frames to RGB for visualization with matplotlib
        frame_rgb = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
        
        ax.imshow(frame_rgb)
        ax.plot(vx[int(k/fps)], vy[int(k/fps)], 'x')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(dir_path, 'temp', str(k+1).zfill(5)+'.png'), dpi=dpi, bbox_inches='tight',pad_inches=0)
        plt.close('all')

video_name = os.path.join(dir_path,'processed','Butterfly','Videos', 'video_' + fs[c] + '.avi')
images = sorted([img for img in  os.listdir(os.path.join(dir_path, 'temp')) if img.endswith(".png")])
frame = cv2.imread(os.path.join(dir_path, 'temp', images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 10.0, (width,height))

for l in range(len(images)):
    video.write(cv2.imread(os.path.join(dir_path, 'temp', images[l])))

cv2.destroyAllWindows()
video.release() 

move_n = np.zeros(len(vx))
for k in range(len(vx)):
    move_n[k] = np.linalg.norm(np.array((vx[k],vy[k]))-np.array((vx[0],vy[0])))

np.savez(os.path.join(dir_path,'processed','Butterfly','Measurements', fs[c] + '_measure.npz'), vx=vx, vy=vy, time=time_f)


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

np.savez(os.path.join(dir_path, 'processed', 'Butterfly', 'Measurements', fs[c] + '_measureIR.npz'), t_max=t_max, time=t)
            
video_name = os.path.join(dir_path,'processed','Butterfly', 'Videos', 'video_ir_' + fs_ir[c] + '.avi')
images = sorted([img for img in os.listdir(os.path.join(dir_path, 'temp')) if img.endswith(".png")])
frame = cv2.imread(os.path.join(dir_path, 'temp', images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 10.0, (width,height))

for l in range(c0):
    video.write(cv2.imread(os.path.join(dir_path, 'temp', images[l])))

cv2.destroyAllWindows()
video.release()

draw_plots(vx, vy, time_f, t, t_max)
plt.savefig(os.path.join(dir_path,'processed','Butterfly','Plots', 'temperature_' + fs[c] + '.pdf'), format='pdf', dpi=dpi)