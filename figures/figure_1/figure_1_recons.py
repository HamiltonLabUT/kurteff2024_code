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

for hem in ['lh','rh']:
	hsubjs = [s for s in subjs if hem in hems[s]]
	patient = img_pipe.freeCoG(subj="cvs_avg35_inMNI152",hem=hem)
	pial, curv = imaging_utils.load_template_brain(hem=hem, inflated=True)
	elecs, anat, all_labels, colors, weights = [], [], [], [], []
	for s in hsubjs:
		pt = img_pipe.freeCoG(f"{s}_complete",subj_dir=data_path,hem=hem)
		e, a = imaging_utils.clip_4mm_elecs(pt, hem=hem, elecfile_prefix="TDT_elecs_all_warped")
		e, a = imaging_utils.clip_outside_brain_elecs(pt,elecmatrix=e,anatomy=a,hem=hem,elecfile_prefix="TDT_elecs_all_warped")
		if len(e) > 0:
			elecs.append(gkip.convert_elecs_to_inflated(pt,e,hem=hem,anat=a,warp=True))
			anat.append(a)
	# Done looping through subjs now, concatenate lists into big arrays for plotting
	if len(elecs) > 0:
		elecs = np.vstack((elecs))
		anat = np.vstack((anat))
		# Plotting params
		azimuth = 175 if hem == 'rh' else -5
		medial_azimuth = -5 if hem == 'rh' else 175
		elec_color = imaging_utils.color_by_roi(anat)
		mesh, mlab = ctmr_gauss_plot(tri=pial['tri'], vert=pial['vert'],
			brain_color=curv, cmap=cm.gray_r, vmin=-2, vmax=8, opacity=1., bgcolor=(1.,1.,1.))
		el_add(elecs, msize=5, labels=None, color=elec_color)
		# Lateral screenshot
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
		plt.savefig(os.path.join(git_path,"figures","figure_1","mayavi_screenshots",
			f"{hem}_lateral.png"),transparent=True)
		mlab.close()
		time.sleep(1)
		GUI().process_events()
		time.sleep(1)
		# Medial screenshot
		mlab.view(azimuth=medial_azimuth, elevation=90, distance=400)
		time.sleep(1)
		GUI().process_events()
		time.sleep(1)
		arr = mlab.screenshot(antialiased=True)
		fig = plt.figure(figsize=(20,20))
		arr, xoff, yoff = remove_whitespace(arr)
		pl.imshow(arr, aspect='equal')
		plt.axis('off')
		plt.tight_layout()
		plt.savefig(os.path.join(git_path,"figures","figure_1","mayavi_screenshots",
			f"{hem}_medial.png"),transparent=True)
		mlab.close()
		time.sleep(1)
		GUI().process_events()
		time.sleep(1)