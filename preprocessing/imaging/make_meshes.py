import os
from img_pipe import img_pipe
import numpy as np

# Path - update locally
data_path = '/path/to/bids/dataset/'

# Subject identifier - update accordingly
subj = "S0004"

outdir = os.path.join(data_path,f"{subj}_complete","label","gyri")
patient = img_pipe.freeCoG(f"{subj}_complete", hem="stereo")
elecs = patient.get_elecs()['elecmatrix']
hems = []
if sum(elecs[:,0] > 0) >= 1:
	hems.append('rh')
if sum(elecs[:,0] < 0) >= 1:
	hems.append('lh')

for hem in hems:
	os.system(
		f"mri_annotation2label --subject {subj}_complete --hemi {hem} --surface pial --annotation aparc.a2009s -sd {data_path} --outdir {outdir}" 
	)
	labels = os.listdir(outdir)
	for label in labels:
		if label[:2] == hem:
			mesh_name = label.replace(f"{hem}.","").replace(".label","")
			patient.make_roi_mesh(mesh_name, [mesh_name], hem=hem)

# Make template meshes as well.
 outdir = os.path.join(data_path, "cvs_avg35_inMNI152", "label", "gyri")
for hem in ['lh','rh']:
	patient = img_pipe.freeCoG("cvs_avg35_inMNI152", hem=hem)
	os.system(
		f"mri_annotation2label --subject cvs_avg35_inMNI152 --hemi {hem} --surface pial --annotation aparc.a2009s --sd {data_path} --outdir {outdir}"
	)
	labels = os.listdir(outdir)
	for label in labels:
		if label[:2] == hem:
			mesh_name = label.replace(f"{hem}.","").replace(".label","")
			patient.make_roi_mesh(mesh_name, [mesh_name], hem=hem)