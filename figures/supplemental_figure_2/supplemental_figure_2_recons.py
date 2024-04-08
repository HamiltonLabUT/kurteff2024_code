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

for s in subjs:
	for hem in hems[s]:

	def plt_single_subject(subj,subj_dir,
	hem='rh', warp=False, inflated=False, clip_outside_brain=True, show_density=False, color_elecs=False,
	device_name=[], single_elec=[], roi = [], bgcolor=(1.,1.,1.), medial=False,
	labels=True, label_scale=4, msize=5, opacity=0.8, gsp=50, savefig=[]):
	'''
	Plots all electrodes (minus subcortical, whitematter, outside brain) for a single subject.
	This function is an amalgamation of many different plotting scripts that I combined on 8/9/23.
	Currently, show_density only works when inflated==False.
	* clip_oustide_brain: bool, defaults to True. If True, excludes electrodes marked as outside the brain in the "IN_BOLT" textfile.
	'''
		patient = img_pipe.freeCoG(subj=f"{s}_complete", hem=hem, subj_dir=data_path)
		pial = patient.get_surf(hem=hem)
		elecs, anat = imaging_utils.clip_4mm_elecs(patient, hem=hem, elecfile_prefix="TDT_elecs_all")
		if len(elecs) > 0:
			elecs, anat = imaging_utils.clip_outside_brain_elecs(patient,elecmatrix=elecs,anatomy=anat,
				hem=hem,elecfile_prefix="TDT_elecs_all")
			azimuth = -5 if hem == 'rh' else 175
			elec_color = imaging_utils.color_by_roi(anat)
			weights = np.ones(elecs.shape[0])
			mesh, mlab = ctmr_gauss_plot(tri=pial['tri'],vert=pial['vert'],opacity=0.8,weights=weights,elecs=elecs,
				gsp=50,vmin=-1,vmax=1,cmap=cm.coolwarm,show_colorbar=False,bgcolor=(0.,0.,0.))
			el_add(elecs, msize=5, labels=None, color=elec_color)
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
			plt.savefig(os.path.join(git_path,"figures","supplemental_figure_2","mayavi_screenshots",
				f"{s}_{hem}_lateral.png"), transparent=True)
			mlab.close()
			time.sleep(1)
			GUI().process_events()
			time.sleep(1)