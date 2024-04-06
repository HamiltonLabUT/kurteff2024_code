# Before you run this script, do the following:
# 1. Convert your T1 MRI scan
# 2. Average across repetitions
# 3. AC-PC (anterior commisure/posterior commisure) alignment in FreeView (do not rely on `acpc_detect`)
# 4. Put the aligned scan into the acpc folder in the BIDS dataset (or just use the one already there!)

# This script does the following:
# 1. Populates the imaging directory with the file structure necessary for img_pipe
# 2. Converts a Curry .pom file to tkRAS coordinates for use in FreeSurfer
# 3. Get the anatomical labels assigned to each electrode
# 4. Warps the electrodes to a template brain (after the electrode labeling in the native space has been manually QC'd)

# Paths, be sure to update this locally
data_path = '/path/to/bids/dataset/'

# Also update the subject you wish to preprocess
subj = 'S0004'

# Imports
from img_pipe import img_pipe
from img_pipe.plotting.ctmr_brain_plot import ctmr_gauss_plot, el_add
import register_pom
from glob import glob
import os
import numpy as np
import nibabel as nib

# Load patient via img_pipe
patient = img_pipe.freeCoG(f"{subj}_complete", hem="stereo")

patient.prep_recon()
patient.get_recon()

print("Navigate to your imaging directly and place your .pom file in the pom directory.")
input("Press Enter to continue...")

# Register the pom file
pom_file = glob(f'{data_path}/{subj}_complete/pom/*.pom')[0]
ch_names, pom_coords, transform, color_nums = register_pom.load_pom(pom_file)
register_pom.plot_pom(ch_names, pom_coords, color_nums)
fsvox2tkras = np.array([[  -1.,    0.,    0.,  128.],
							[   0.,    0.,    1., -128.],
							[   0.,   -1.,    0.,  128.],
							[   0.,    0.,    0.,    1.]])
subset_list = ['LOCC3', 'LCMN12', 'LPCUN9', 'RSPL4']
CRS_subset = np.array(
				[[144.39, 130.28, 39.13],
				 [170.65, 85.09, 150.69],
				 [128.33, 78.59, 56.37],
				 [72.23, 100.4, 75.06]],
			)
ch_types = ['depth']*len(ch_names)
tdt_elecs_all, b = register_pom.align_from_subset(
	subj, pom_coords, ch_names, subset_list, CRS_subset, ch_types, data_path)
brain_mri = f'{data_path}/{subj}_complete/mri/brain.mgz'
elecs_CT = f'{data_path}/{subj}_complete/CT/rCT.nii'
brain=nib.load(brain_mri)
vox2ras = brain.affine
tdt_elecs_ones = np.hstack((tdt_elecs_all,  np.ones((tdt_elecs_all.shape[0],1))))
tdt_elecs_RAS = np.dot(vox2ras, np.dot(np.linalg.inv(fsvox2tkras) , tdt_elecs_ones.T))[:3,:].T
RAS_check_points = f'{data_path}/{subj}_complete/elecs/TDT_RAS_check.txt'
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

print("Quality checking the pial surface")
patient.check_pial()
input("Press Enter to continue...")

# Get surface and subcortical meshes
patient.convert_fsmesh2mlab()
patient.get_subcort()

# Co-register CT to T1 scan and identify electrodes on coregistered CT
patient.reg_img()
print("Marking electrodes on coregistered CT")
patient.mark_electrodes()
input("Press Enter to continue...")

interp = input("Interpolate grid corners? (If necessary) (type y/n)")
if interp == 'y':
	patient.interp_grid()

# Project electrodes to mesh surface, then create the montage
patient.project_electrodes()
patient.make_elecs_all()

# Anatomically label electrodes according to Destrieux atlas
patient.label_elecs()

print("Quality-checking anatomy")
patient.plot_recon_anatomy()
input("Press Enter to continue...")

edit_labels = input("Edit mislabeled electrodes? (type y/n)")
if edit_labels == 'y':
	patient.edit_elesc_all()

print("Now, open the CT, brain.mgz, and apart.a2009s+aseg.nii in FreeView.")
print("Manually check any electrodes with the label 'Unknown' and adjust accordingly.")
print("Only electrodes outside the brain (e.g., in the bolt, in the skull) should retain that anatomical label.")
input("Press Enter when you have completed this step...")

# Warp to template brain
patient.warp_all()

# Plot the warp for QC
template = img_pipe.freeCoG("cvs_avg35_inMNI152", hem="stereo")
pial = template.get_surf(hem="stereo")
elecs = patient.get_elecs(elecfile_prefix="TDT_elecs_all_warped")['elecmatrix']
labels = [a[0][0] for a in patient.get_elecs(elecfile_prefix="TDT_elecs_all_warped")['anatomy']]
mesh, mlab = ctmr_gauss_plot(tri=pial['tri'], vert=pial['vert'], opacity=0.8)
el_add(elecs,labels=labels)
mlab.view()
input("Press Enter when you are finished checking the warp...")
