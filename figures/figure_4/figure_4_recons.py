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

subjs = [s for s in os.listdir(
	os.path.join(git_path,"preprocessing","events","csv")) if "TCH" in s or "S0" in s]
exclude = ["TCH8"]
no_imaging = ["S0010"]
subjs = [s for s in subjs if s not in exclude and s not in no_imaging]

hems = {s:[] for s in subjs}
for s in subjs:
	pt = img_pipe.freeCoG(f"{s}_complete",hem='stereo',subj_dir=data_path)
	elecs = pt.get_elecs()['elecmatrix']
	if sum(elecs[:,0] > 0) >= 1:
		hems[s].append('rh')
	if sum(elecs[:,0] < 0) >= 1:
		hems[s].append('lh')

# read pandas
recon_types = ['within_clust_1','within_clust_2','within_clust_3','across_clust_1-2']
figure_dfs = [pd.read_csv(os.path.join(git_path,"figures","figure_4","csv","figure_4_cmap_within_clust_1.csv")),
              pd.read_csv(os.path.join(git_path,"figures","figure_4","csv","figure_4_cmap_within_clust_2.csv")),
              pd.read_csv(os.path.join(git_path,"figures","figure_4","csv","figure_4_cmap_within_clust_3.csv")),
              pd.read_csv(os.path.join(git_path,"figures","figure_4","csv","figure_4_cmap_across_clust_1-2.csv"))]

for i,figure_df in enumerate(figure_dfs):
	n_subjs = np.unique(np.array(figure_df['subj'].values)).shape[0]
	for hem in ['lh','rh']:
		hem_df = figure_df.loc[figure_df['hem']==hem]
		elecs = np.vstack((hem_df['x'],hem_df['y'],hem_df['z'])).T
		colors = np.vstack((hem_df['r'],hem_df['g'],hem_df['b'])).T
		# load template
		pial, curv = imaging_utils.load_template_brain(hem=hem, inflated=True)
		# Plot
		azimuth = -5 if hem == 'rh' else 175
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
		plt.savefig(os.path.join(git_path,"figures","figure_4","mayavi_ss",
			f"{hem}_recon_{n_subjs}_{recon_types[i]}.png"), transparent=True)
		mlab.close()
		time.sleep(1)
		GUI().process_events()
		time.sleep(1)
		print(hem, "Screenshot saved")