from img_pipe import img_pipe
from img_pipe.plotting.ctmr_brain_plot import ctmr_gauss_plot
from img_pipe.plotting.ctmr_brain_plot import el_add
import nibabel
import numpy as np
import pandas as pd
import os
from PIL import ImageColor
import warnings
from pyface.api import GUI
import pylab as pl
from img_pipe.SupplementalFiles import FS_colorLUT
from img_pipe.img_pipe import remove_whitespace
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import time
import warnings

def condense_roi(fs_roi):
	'''
	An ROI compression scheme borrowed from Maansi Desai's code.
	I rewrote it to save space but I preserved all her
	freesurfer label-to-condensed label associations where possible.
	The one exception is the insula, where I split it up into more fine-grained labels.
	I also added more labels that are present in my dataset but not hers, trying as best
	as possible to preserve her initial categorization.
	'''
	condensed_rois = {
		# Subcort/WM
		'cerebellum' : ['Cerebellum-Cortex'],
		'amyg' : ['Amygdala'],
		'hippo' : ['G_oc-temp_med-Parahip','Hippocampus'],
		'thal' : ['Thalamus'],
		'cing' : ['G_subcallosal','G_and_S_cingul-Mid-Post','G_cingul-Post-dorsal','G_cingul-Post-ventral','G_and_S_cingul-Mid-Ant','G_and_S_cingul-Ant','S_subparietal'],
		'striatum' : ['Putamen','Caudate'],
		'wm' : ['Cerebral-White-Matter','Cerebral-Cortex','CC_Anterior','Cerebral-White-Cortex','S_pericallosal','WM-hypointensities'],
		'outside_brain' : ['Unknown','unknown','Inf-Lat-Vent','Lateral-Ventricle','VentralDC'],
		# Frontal
		'SFG' : ['superiorfrontal','G_front_sup','ctx_lh_superiorfrontal','ctx_rh_superiorfrontal'],
		'SFS' : ['S_front_sup'],
		'MFG' : ['front_middle','caudalmiddlefrontal','G_front_middle','rostralmiddlefrontal'],
		'MFS' : ['S_front_middle'],
		'IFG' : ['G_front_inf-Orbital','G_front_inf-Triangul','G_front_inf-Opercular','parsopercularis'],
		'IFS' : ['S_front_inf','S_precentral_inf-part'],
		'OFC' : ['S_suborbital','S_orbital_lateral','S_orbital_med-olfact','S_orbital-H_Shaped','G_orbital'],
		'front_operculum' : ['Lat_Fis-ant-Horizont','Lat_Fis-ant-Vertical'],
		'front_pole' : ['G_rectus','G_and_S_transv_frontopol'],
		# Precentral
		'preCG' : ['G_precentral','precentral'],
		'preCS' : ['S_precentral', 'S_precentral-sup-part','S_precentral-inf-part','S_central'],
		# Postcentral
		'postCG' : ['G_postcentral','postcentral'],
		'postCS' : ['S_postcentral'],
		# Paracentral
		'paracentral' : ['G_and_S_paracentral'],
		# Subcentral
		'subcentral' : ['G_and_S_subcentral'],
		# Temporal
		'STG' : ['G_temp_sup-Lateral','superiortemporal','G_temporal_sup-Lateral'],
		'STS' : ['S_temporal_sup','bankssts'],
		'MTG' : ['G_temporal_middle','middletemporal'],
		'ITG' : ['G_temporal_inf'],
		'ITS' : ['S_temporal_inf'],
		'PT' : ['G_temp_sup-Plan_tempo'],
		'PP' : ['G_temp_sup-Plan_polar'],
		'HG' : ['G_temporal_transverse','G_temp_sup-G_T_transv','S_temporal_transverse'],
		'temp_pole' : ['Pole_temporal'],
		# Parietal
		'supramar' : ['G_pariet_inf-Supramar','supramarginal'],
		'angular' : ['G_pariet_inf-Angular'],
		'SPL' : ['superiorparietal','G_parietal_sup','G_precuneus','S_parieto_occipital'],
		'IPL' : ['inferiorparietal'],
		'intraparietal' : ['S_interm_prim-Jensen','S_intrapariet_and_P_trans'],
		# Insula
		'insula_inf' : ['S_circular_insula_inf'],
		'insula_sup' : ['S_circular_insula_sup'],
		'insula_post' : ['G_Ins_lg_and_S_cent_ins','Lat_Fis-post'],
		'insula_ant' : ['S_circular_insula_ant','G_insular_short'],
		# Occipital (I'm not very good at occipital lobe anatomy so some of this may be off...)
		'cuneus' : ['cuneus','G_cuneus','S_oc_sup_and_transversal'],
		'lunate' : ['S_oc_middle_and_Lunatus'],
		'lingual' : ['S_oc-temp_med_and_Lingual','G_oc-temp_med-Lingual'],
		'calcarine' : ['S_calcarine'],
		'occ_temp' : ['G_oc-temp_lat-fusifor','S_oc-temp_lat','S_collat_transv_ant'],
		'occ_pole' : ['G_occipital_middle','Pole_occipital','G_occipital_sup','G_and_S_occipital_inf']
	}
	# Remove information that's redundant across hemispheres
	fs_roi_strip = fs_roi.replace("ctx_rh_","").replace("ctx_lh_","").replace("Left-","").replace("Right-","")
	try:
		condensed_roi = [r for r in condensed_rois.keys() if fs_roi_strip in condensed_rois[r]][0]
	except:
		print(fs_roi_strip)
		raise Exception(f'ROI {fs_roi_strip} missing from condensed_rois dict, please add it')
	return condensed_roi

def clip_hem_elecs(patient,hem='rh',elecfile_prefix="TDT_elecs_all",
	elecmatrix=None,anatomy=None, return_idxs=False):
	'''
	Clips electrodes according to hemisphere. Returns elecmatrix and anatomy.
	'''
	if elecmatrix is None and anatomy is None:
		print("No elecmatrix/anat specified; clipping from all patient elecs...")
		all_xyz = patient.get_elecs(elecfile_prefix=elecfile_prefix)['elecmatrix']
		all_anat = patient.get_elecs(elecfile_prefix=elecfile_prefix)['anatomy']
	else:
		print("Clipping from specified subset of patient elecs...")
		all_xyz = elecmatrix
		all_anat = anatomy
	elecmatrix, anatomy, idxs = [], [], []
	for i,e in enumerate(all_xyz):
		if hem == 'rh':
			# Positive X only
			in_hem = e[0] >= 0
		else:
			# Negative X only
			in_hem = e[0] <= 0
		if in_hem:
			if hem == 'rh':
				if '_lh_' in all_anat[i][3][0] or 'left' in all_anat[i][3][0].lower():
					# Sometimes a positive X coordinate is still a LH elec because it crosses the midline
					# This and the following if statements relating to skip_append check
					# for that to prevent false positive errors.
					skip_append = True
				else:
					skip_append = False
			if hem == 'lh':
				if '_rh_' in all_anat[i][3][0] or 'right' in all_anat[i][3][0].lower():
					skip_append = True
				else:
					skip_append = False
			if not skip_append:
				elecmatrix.append(e)
				anatomy.append(all_anat[i])
				idxs.append(i)
	if return_idxs:
		return np.array(elecmatrix),np.array(anatomy), idxs
	else:
		return np.array(elecmatrix),np.array(anatomy)

def gross_anat(fs_roi):
	'''
	Condenses freesurfer labels into a set of manageable ROIs for visualization/analysis:
	frontal, temporal, parietal, occipital, precentral, postcentral, insula, subcortical, and white matter.
	'''
	condensed_rois = {
		'frontal' : ['superiorfrontal','caudalmiddlefrontal','S_suborbital','S_orbital_lateral','S_orbital_med-olfact','S_orbital-H_Shaped','G_orbital',
					 'S_front_inf','S_front_middle','S_front_sup','G_front_inf-Opercular','G_front_inf-Orbital', 'parsopercularis','rostralmiddlefrontal',
					 'G_front_inf-Triangul','G_front_middle','G_front_sup','Lat_Fis-ant-Horizont','Lat_Fis-ant-Vertical', 'ctx_rh_S_precentral_inf-part',
					 'ctx_lh_G_rectus','ctx_rh_G_rectus','ctx_rh_G_and_S_transv_frontopol','ctx_lh_G_and_S_transv_frontopol', 'ctx_lh_S_precentral_inf-part',
					 'ctx_lh_superiorfrontal','ctx_rh_superiorfrontal'],
		'temporal' : ['S_temporal_inf','S_temporal_sup','S_temporal_transverse','S_collat_transv_ant',
					  'G_temp_sup-G_T_transv','G_temp_sup-Lateral','G_temp_sup-Plan_polar','G_temp_sup-Plan_tempo',
					  'G_temporal_middle','G_temporal_sup-Lateral','Pole_temporal','ctx_lh_G_temporal_inf',
					  'ctx_rh_G_temporal_inf','superiortemporal','middletemporal','bankssts'],
		'parietal' : ['superiorparietal','inferiorparietal','supramarginal','S_interm_prim-Jensen','G_pariet_inf-Angular','G_pariet_inf-Supramar','G_parietal_sup',
					  'ctx_lh_S_intrapariet_and_P_trans','ctx_rh_S_intrapariet_and_P_trans','ctx_lh_G_precuneus','ctx_rh_G_precuneus',
					  'ctx_lh_S_parieto_occipital','ctx_rh_S_parieto_occipital'],
		'occipital' : ['cuneus','S_oc_middle_and_Lunatus','S_oc-temp_med_and_Lingual','S_calcarine','G_oc-temp_lat-fusifor',
					   'G_oc-temp_med-Lingual','G_occipital_middle','Pole_occipital','ctx_lh_G_cuneus','ctx_rh_G_cuneus',
					   'ctx_lh_G_occipital_sup','ctx_rh_G_occipital_sup','ctx_lh_G_and_S_occipital_inf','ctx_rh_G_and_S_occipital_inf',
					   'ctx_lh_S_oc-temp_lat','ctx_rh_S_oc-temp_lat', 'ctx_lh_S_oc_sup_and_transversal', 'ctx_rh_S_oc_sup_and_transversal'],
		'precentral' : ['precentral','S_precentral-inf-part','S_central','G_precentral','G_and_S_subcentral','ctx_lh_S_precentral-sup-part',
						'ctx_rh_S_precentral-sup-part','precentral'],
		'postcentral' : ['postcentral','S_postcentral','G_postcentral','ctx_rh_G_and_S_paracentral','ctx_lh_G_and_S_paracentral','postcentral'],
		'insula' : ['S_circular_insula_ant','S_circular_insula_inf','S_circular_insula_sup',
					'G_Ins_lg_and_S_cent_ins','G_insular_short','Lat_Fis-post'],
		'subcort' : ['Right-Cerebellum-Cortex','Right-Hippocampus','Right-Amygdala','Left-Hippocampus','Left-Amygdala','Left-Caudate',
					 'G_and_S_cingul-Ant','G_and_S_cingul-Mid-Ant','G_oc-temp_med-Parahip','S_subparietal',
					 'ctx_lh_G_cingul-Post-ventral','ctx_rh_G_cingul-Post-ventral','ctx_lh_G_cingul-Post-dorsal',
					 'ctx_rh_G_cingul-Post-dorsal','Left-Thalamus','Right-Thalamus','ctx_lh_G_and_S_cingul-Mid-Post',
					 'ctx_rh_G_and_S_cingul-Mid-Post','Left-Putamen','Right-Putamen','ctx_rh_G_subcallosal','ctx_lh_G_subcallosal'],
		'whitematter' : ['ctx-rh-unknown','Unknown','WM-hypointensities','S_pericallosal','Right-Cerebral-White-Cortex',
						 'Left-Cerebral-White-Matter','Right-Cerebral-White-Matter','Left-Cerebral-White-Cortex',
						 'Left-Cerebral-Cortex','Right-Cerebral-Cortex','CC_Anterior'],
		'outside_brain' : ['Right-Inf-Lat-Vent','Left-Lateral-Ventricle','Right-Lateral-Ventricle','Left-Inf-Lat-Vent','Left-VentralDC','Right-VentralDC']
	}
	try:
		condensed_roi = [r for r in condensed_rois.keys() if fs_roi[7:] in condensed_rois[r]][0]
	except:
		try:
			condensed_roi = [r for r in condensed_rois.keys() if fs_roi in condensed_rois[r]][0]
		except:
			print(fs_roi)
			print(fs_roi[7:])
			raise Exception(f'ROI {fs_roi} missing from condensed_rois dict, please add it')
	return condensed_roi