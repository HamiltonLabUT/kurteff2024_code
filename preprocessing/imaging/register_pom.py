import numpy as np
import sys
import os
import matplotlib
matplotlib.use('macosx')
from matplotlib import pyplot as plt
from matplotlib import cm
import nibabel as nib
import scipy.cluster.vq
import glob
import scipy.io

#pip install simpleicp  
#from simpleicp import PointCloud, SimpleICP

def load_pom(pom_file):
	'''
	This function loads a .pom file and returns the channel names,
	3D coordinates, and transform

	Inputs:
		pom_file [str] : the path to the .pom file

	Outputs:
		ch_names [list] : list of channel names
		pom_coords [array] : array of coordinates [nchans x 3] from the pom file
		transform [array] : 4 x 4 transform matrix from pom file
		color_nums [list] : list of color numbers for each electrode (same color
							for the same device)
	'''

	ch_names = []
	pom_coords = []
	color_nums = []
	with open(pom_file, 'r') as f:
		text_contents = f.readlines()
		for idx, line in enumerate(text_contents):
			if 'POINT_TRAFO START_LIST' in line:
				print(f'Found transform matrix on line {idx}')
				# Read the next four lines
				transform = []
				for i in np.arange(1,5):
					transform.append([float(t) for t in text_contents[idx+i].split()])
				print('Transform matrix: ')
				print(transform)
			if 'LOCATION_LIST START_LIST' in line:
				print(f'Found locations on line {idx}')
				pom_coords = []
				curr_idx = idx+1
				while 'LOCATION_LIST END_LIST' not in text_contents[curr_idx]:
					pom_coords.append([float(t) for t in text_contents[curr_idx].split()])
					curr_idx+=1
				print(f'Found {len(pom_coords)} total electrodes in the pom file')
				pom_coords = np.array(pom_coords, dtype=float)

			if 'REMARK_LIST START_LIST' in line:
				print(f'Found channel names on line {idx}')
				ch_names = []
				curr_idx = idx+1
				while 'REMARK_LIST END_LIST' not in text_contents[curr_idx]:
					ch_names.append(text_contents[curr_idx].split()[0])
					curr_idx+=1
				print(f'Found {len(ch_names)} total channel names in the pom file')
				print(ch_names)

			if 'COLOR_LIST START_LIST' in line:
				print(f'Found colors on line {idx}')
				color_nums = []
				curr_idx = idx+1
				while 'COLOR_LIST END_LIST' not in text_contents[curr_idx]:
					color_nums.append(int(text_contents[curr_idx].split()[0]))
					curr_idx+=1
				print(f'Found {len(color_nums)} total colors in the pom file')
				print(color_nums)

	print('Flipping pom coordinates since theyre in LPS and we want RAS eventually')
	#pom_coords[:,1] = -pom_coords[:,1]
	#pom_coords[:,0] = -pom_coords[:,0]


	return ch_names, pom_coords, transform, color_nums


def plot_pom(ch_names, pom_coords, color_nums, ax=None):
	'''
	Inputs:
		ch_names [list] : list of channel names
		pom_coords [array] : array of coordinates [nchans x 3] from the pom file
		color_nums [list] : list of color numbers for each electrode (same color
							for the same device)

	Outputs:
		fig : figure handle

	'''
	plt.ion()
	color_idx = dict(zip(np.unique(color_nums), np.arange(len(np.unique(color_nums)))))
	new_cmap = cm.jet(np.linspace(0,1,len(np.unique(color_nums))))
	elec_colors = [new_cmap[color_idx[i]] for i in color_nums]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(pom_coords[:,0], pom_coords[:,1], pom_coords[:,2], s=20, c=elec_colors)
	for elec in np.arange(pom_coords.shape[0]):
		ax.text(pom_coords[elec,0], pom_coords[elec,1], pom_coords[elec,2], ch_names[elec], fontdict={'fontsize': 'xx-small'})
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()

	return fig


def get_electrode_centroids(subj, ch_names, ct_file, imaging_dir = '/Users/liberty/Desktop', thresh=250, nk=None):
	'''
	Inputs:
		subj [str] : the subject ID
		ch_names [list] : list of channel names from the pom file
		ct_file [str] : full path to the CT file
		imaging_dir [str] : your imaging directory
		thresh [int] : threshold for finding electrodes in the CT or MRI
		nk [int] : number of clusters (defaults to len(ch_names))
	Outputs:
		centroids [array] : matrix of [nchans x 3] coordinates for electrodes (in CRS) from kmeans
		distortion [float] : The mean (non-squared) Euclidean distance between the observations
							 passed and the centroids generated from kmeans clustering.
	'''
	if nk is None:
		nk = len(ch_names)
	bin_ct = f'{imaging_dir}/{subj}/pom/{subj}_thr{thresh}_1mm_iso.nii.gz'
	thresh_ct = f'{imaging_dir}/{subj}/pom/{subj}_thr{thresh}.nii.gz'

	if not os.path.isfile(bin_ct):
		print('Binarizing the rCT scan')
		fsl_cmd = f'/usr/local/fsl/bin/fslmaths {ct_file} -thr {thresh} {thresh_ct}'
		os.system(fsl_cmd)

		print('Reslicing to 1 mm isotropic voxel size')
		flirt_cmd = f'/usr/local/fsl/bin/flirt -in {thresh_ct} -ref {thresh_ct} -applyisoxfm 1.0 -nosearch -out {bin_ct}'
		os.system(flirt_cmd)

	print('Loading 1 mm isotropic binarized CT scan')
	a = nib.load(bin_ct)
	inds = np.where(a.get_fdata()>0)
	elec_coords = np.vstack((inds)).T.astype(float)

	if nk>0:
		print('Clustering data to get the electrodes')
		centroids, distortion = scipy.cluster.vq.kmeans(elec_coords, nk)
	else:
		centroids = []
		distortion = 0
	return centroids, distortion, elec_coords


def run_icp(centroids, pom_coords, min_planarity=0.1):
	'''
	Run iterative closest point algorithm... doesn't quite work yet
	'''
	print('Mean-centering the CT centroids and pom cordinates for alignment')
	centroids_translated = (centroids - centroids.mean(0))
	pom_translated = (pom_coords - (pom_coords.mean(0)-centroids_translated.mean(0)))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(centroids_translated[:,0], centroids_translated[:,1], centroids_translated[:,2], s=20, label='CT')
	ax.set_box_aspect((1, 1, 1))
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Original centroids before alignment')
	plt.legend()
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(pom_translated[:,0], pom_translated[:,1], pom_translated[:,2], s=20, c='r', label='pom')
	ax.set_box_aspect((1, 1, 1))
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Original centroids before alignment')
	plt.legend()
	plt.show()

	# Create point cloud objects
	print('Initializing point clouds')
	pc_fix = PointCloud(centroids_translated.copy(), columns=["x", "y", "z"])
	pc_mov = PointCloud(pom_translated.copy(), columns=["x", "y", "z"])


	# Create simpleICP object, add point clouds, and run algorithm!
	print(f'Running iterative closest point (ICP) algorithm with min_planarity={min_planarity}')
	icp = SimpleICP()
	icp.add_point_clouds(pc_fix, pc_mov)
	H, X_mov_transformed, rigid_body_transformation_params = icp.run(min_planarity=min_planarity, max_iterations=100)

	print('Shifting coordinates back to original centroid position')
	new_coords = X_mov_transformed + centroids.mean(0)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], s=20, label='CT')
	ax.scatter(new_coords[:,0], new_coords[:,1], new_coords[:,2], s=20, c='k', label='aligned pom')
	plt.title('Centroids and pom coordinates after alignment')
	plt.legend()
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(centroids_translated[:,0], centroids_translated[:,1], centroids_translated[:,2], s=20, label='CT')
	ax.scatter(X_mov_transformed[:,0], X_mov_transformed[:,1], X_mov_transformed[:,2], s=20, c='k', label='aligned pom')
	plt.title('Centroids and pom coordinates after alignment no re-translate')
	plt.legend()
	plt.show()

	return new_coords


def align_from_subset(subj, pom_coords, ch_names, subset_list, CRS_subset, ch_types, imaging_dir):
	'''
	Inputs: 
		pom_coords [array] : pom coordinates from the file
		subset_list [list] : list of four electrodes you've identified in the MRI
		CRS_subset [array] : coordinates for the four electrodes (CRS) in the MRI
	Outputs:
		tdt_elecs_all [array] : coordinates in TkRAS surface space

	'''

	CRS = np.hstack((CRS_subset, np.ones((CRS_subset.shape[0],1))))
	
	pom_ones = np.hstack((pom_coords, np.ones((pom_coords.shape[0],1))))
	pom = np.array([pom_ones[ch_names.index(c),:] for c in subset_list])

	b = np.dot(CRS.T, np.linalg.inv(pom.T))
	tdt_elecs_CRS = np.dot(b, pom_ones.T)[:3,:].T

	fsvox2tkras = np.array([[  -1.,    0.,    0.,  128.],
                   			[   0.,    0.,    1., -128.],
                   			[   0.,   -1.,    0.,  128.],
                   			[   0.,    0.,    0.,    1.]])

	tdt_elecs_all = np.dot(fsvox2tkras, np.dot(b, pom_ones.T))[:3,:].T

	neweleclabels = np.empty((tdt_elecs_all.shape[0], 3), dtype=object)
	neweleclabels[:,0] = ch_names
	neweleclabels[:,1] = ch_names
	neweleclabels[:,2] = ch_types
	
	elecs_file = f'{imaging_dir}/{subj}/elecs/TDT_elecs_all.mat'
	print(f'Saving electrodes to {elecs_file}')
	scipy.io.savemat(elecs_file, {'elecmatrix': tdt_elecs_all,'eleclabels': neweleclabels})
	
	return tdt_elecs_all, b


if True:
	imaging_dir = '/Users/jsh3653/Library/CloudStorage/Box-Box/NIH_ECoG_imaging/TCH_imaging'
	#imaging_dir = '/Users/liberty/Library/CloudStorage/Box-Box/ECoG_imaging'
	subj = 'TCH19'
	pom_file=glob.glob(f'{imaging_dir}/{subj}/pom/*.pom')[0]
	print(pom_file)
	ch_names, pom_coords, transform, color_nums = load_pom(pom_file)
	plot_pom(ch_names, pom_coords, color_nums)

	fsvox2tkras = np.array([[  -1.,    0.,    0.,  128.],
                   			[   0.,    0.,    1., -128.],
                   			[   0.,   -1.,    0.,  128.],
                   			[   0.,    0.,    0.,    1.]])
	#ct_file = f'{imaging_dir}/{subj}/pom/T1_CURRY_elecs_acpc.nii.gz'
	#centroids, distortion, elec_coords = get_electrode_centroids(subj, ch_names, ct_file, imaging_dir=imaging_dir, thresh=252, nk=len(ch_names))
	#new_coords = run_icp(centroids, pom_coords, min_planarity=0.6)

	# TCH6:
	#flirt -in TCH8_CURRY_8_Saved_Image_Data_Localize_2a.nii.gz -ref ../../mri/T1.nii -out TCH8_CURRY_8_Saved_Image_Data_Localize_2a_acpc.nii.gz -v
	# subset_list = ['LOT14','RFP1','RSH4','ROT14']
	# CRS_subset = np.array(
	# 				[[160.28, 113.17, 62.8],
	# 				[114.88, 120.02, 187.1],
	# 				[104.24, 65.58, 94.48],
	# 				[103.65, 110.77, 69.66]],
	# 			)

	# ch_types = ['depth']*len(ch_names)
	# tdt_elecs_all, b = align_from_subset(subj, pom_coords, ch_names, subset_list, CRS_subset, ch_types, imaging_dir)

	# # TCH7:
	# subset_list = ['RFP1','RANT4','LOT14','RSMA6']
	# CRS_subset = np.array(
	# 				[[124.07, 119.65, 186.36],
	# 				[72.28, 141.3, 117.9],
	# 				[166.85, 123.3, 63.22],
	# 				[116.75, 48.1, 112.47]],
	# 			)

	# # TCH8 Phase 2a:
	# #flirt -in TCH8_CURRY_8_Saved_Image_Data_Localize_2a.nii.gz -ref ../../mri/T1.nii -out TCH8_CURRY_8_Saved_Image_Data_Localize_2a_acpc.nii.gz -v
	# subset_list = ['LFP1','LANT12','RMF4','RFP12']
	# CRS_subset = np.array(
	# 				[[133.38, 153.78, 182.81],
	# 				[200.94, 149.63, 108.53],
	# 				[77.33, 96.42, 138.96],
	# 				[122.03, 93.91, 192.56]],
	# 			)

	# # TCH2:
	# subset_list = ['ROT14','LOT1','RFP10','LACING1']
	# CRS_subset = np.array(
	# 				[[99.17, 120.9, 80.71],
	# 				[144.14, 143.77, 137.02],
	# 				[119.6, 84.3, 194.26],
	# 				[129.51, 141.67, 167.01]],
	# 			)

	# # TCH3:
	# subset_list = ['LOT1','ROT14','ROFGS1','LSMA5']
	# CRS_subset = np.array(
	# 				[[156.06, 137.98, 112.51],
	# 				 [110, 110.83, 58.74],
	# 				 [97.48, 131.45, 174.5],
	# 				 [135.75, 55.29, 114.04]],
	# 			)
	# ch_types = ['depth']*len(ch_names)
	# tdt_elecs_all, b = align_from_subset(subj, pom_coords, ch_names, subset_list, CRS_subset, ch_types, imaging_dir)

	# # TCH5:
	# subset_list = ['RSPL6','LANT10','LFP12','LPCUN9']
	# CRS_subset = np.array(
	# 				[[76.97, 81.98, 83.43],
	# 				 [197.02, 146.75, 109],
	# 				 [145.26, 87.52, 186.17],
	# 				 [141.97, 95.27, 64.57]],
	# 			)

	# S0024
	# subset_list = []
	# CRS_subset = np.array(
	# 				[[],
	# 				 [],
	# 				 [],
	# 				 []],
	# 			)

	# # TCH13
	# subset_list = ['LOT14', 'LANT1', 'LPULV14', 'RSMA4']
	# CRS_subset = np.array(
	# 				[[156.42, 130, 88.29],
	# 				 [160.13, 155, 153.73],
	# 				 [98.09, 80.96, 82.53],
	# 				 [116.82, 55.72, 99.48]],
	# 			)

	# TCH14
	# subset_list = ['RANT1', 'LANT12', 'LAISG14', 'LPCUN5']
	# CRS_subset = np.array(
	# 				[[76.94, 145.55, 150.65],
	# 				 [189.99, 124.92, 114.04],
	# 				 [126.07, 72.66, 186.28],
	# 				 [145.9, 78.24, 72.27]],
	# 			)

	# TCH19
	subset_list = ['LOCC3', 'LCMN12', 'LPCUN9', 'RSPL4']
	CRS_subset = np.array(
					[[144.39, 130.28, 39.13],
					 [170.65, 85.09, 150.69],
					 [128.33, 78.59, 56.37],
					 [72.23, 100.4, 75.06]],
				)
	ch_types = ['depth']*len(ch_names)
	tdt_elecs_all, b = align_from_subset(subj, pom_coords, ch_names, subset_list, CRS_subset, ch_types, imaging_dir)

	brain_mri = f'{imaging_dir}/{subj}/mri/brain.mgz'
	elecs_CT = f'{imaging_dir}/{subj}/CT/rCT.nii'
	
	brain=nib.load(brain_mri)
	vox2ras = brain.affine

	tdt_elecs_ones = np.hstack(( tdt_elecs_all,  np.ones((tdt_elecs_all.shape[0],1)) ))
	tdt_elecs_RAS = np.dot(vox2ras, np.dot(np.linalg.inv(fsvox2tkras) , tdt_elecs_ones.T))[:3,:].T

	RAS_check_points = f'{imaging_dir}/{subj}/elecs/TDT_RAS_check.txt'
	with open(RAS_check_points, 'w') as f:
		for i in np.arange(tdt_elecs_RAS.shape[0]):
			print(i)
			line = '%.6f %.6f %.6f\n'%(tdt_elecs_RAS[i,0], tdt_elecs_RAS[i,1], tdt_elecs_RAS[i,2])
			print(line)
			f.write(line)
		f.write('info\n')
		f.write(f'numpoints {tdt_elecs_RAS.shape[0]}\n')
		f.write('useRealRAS 1')
	os.system(f"freeview --volume {brain_mri}:opacity=0.8 --volume {elecs_CT}:opacity=0.6:colormap=heat:isosurface=1500,3000 --ras 0 0 0 --control-points {RAS_check_points}:radius=2 --viewport 'coronal'")


	# The IDEA

	# We need the Axial image with the identified electrodes from CURRY
	# as bright pixels
	# Make sure the axial image is in the same space as T1.mgz through FLIRT
	# Find the centroids from the axial image
	# 
	# flirt -in TCH7_20220920185306_CURRY_8_Saved_Image_Data_Localize_11038586.nii.gz -ref PreOpMRI/T1_acpc.nii -out T1_CURRY_elecs_acpc.nii -v

	# flirt -in TCH7_20220920185306_CURRY_8_Saved_Image_Data_Localize_11038586.nii.gz -ref /Users/liberty/Library/CloudStorage/Box-Box/NIH_ECoG_imaging/TCH_imaging/TCH7/mri/T1.nii -out T1_CURRY_elecs_acpc2.nii -v

