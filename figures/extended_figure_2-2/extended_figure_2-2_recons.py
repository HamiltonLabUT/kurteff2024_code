# Paths - Update locally!
git_path = '/path/to/git/kurteff2024_code/'
data_path = '/path/to/bids/dataset/'

import os
import numpy as np
import pandas as pd
import nibabel
from pyface.api import GUI
import pylab as pl
from img_pipe import img_pipe
from img_pipe.plotting.ctmr_brain_plot import ctmr_gauss_plot
from img_pipe.plotting.ctmr_brain_plot import el_add
from img_pipe.SupplementalFiles import FS_colorLUT
from img_pipe.img_pipe import remove_whitespace
from PIL import ImageColor
import warnings
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import time
import warnings
import sys
sys.path.append(os.path.join(git_path,"preprocessing","imaging"))
import imaging_utils

subj = "S0007"

hem = 'rh'

# read pandas
figure_df = pd.read_csv(os.path.join(git_path,"figures","figure_2","csv","figure_2_cmap.csv"))
hem_df = figure_df.loc[(figure_df['hem']==hem)&(figure_df['subj']==subj)]
elecs = np.vstack((hem_df['x'],hem_df['y'],hem_df['z'])).T
colors = np.vstack((hem_df['r'],hem_df['g'],hem_df['b'])).T
# Load the mesh
patient = img_pipe.freeCoG(subj=f"{subj}_complete", hem=hem, subj_dir=data_path)
pial = patient.get_surf(hem=hem, roi="inflated")
curv = imaging_utils.load_curvature(patient, hem)
curv[np.where(curv<=0)[0]] = -1
curv[np.where(curv>0)[0]] = 1
# Plot
azimuth = -5
mesh, mlab = ctmr_gauss_plot(tri=pial['tri'],vert=pial['vert'],
					   brain_color=curv, cmap=cm.gray_r, vmin=-2, vmax=8, opacity=1., bgcolor=(1.,1.,1.))
el_add(elecs, color=colors, msize=5, labels=None)
# Save screenshot
mlab.view(azimuth=azimuth, elevation=90, distance=400)
time.sleep(1)
GUI().process_events()
time.sleep(1)
arr = mlab.screenshot(antialiased=True)
fig = plt.figure(figsize=(20,20))
arr, xoff, yoff = remove_whitespace(arr)
pl.imshow(arr, aspect='equal')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(git_path,"figures","extended_figure_2-2","mayavi_ss",
f"{subj}_{hem}_recon.png"), transparent=True)
mlab.close()
time.sleep(1)
GUI().process_events()
time.sleep(1)
print(hem, "Screenshot saved")