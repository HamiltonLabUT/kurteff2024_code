from img_pipe import img_pipe

# Path - update locally
data_path = '/path/to/bids/dataset/'

# Subject identifier - update accordingly
subj = "S0004"

for hem in ['lh','rh','stereo']:
	patient = img_pipe.freeCoG(subj=f"{subj}_complete",hem=hem)
	patient.convert_fsmesh2mlab(mesh_name='inflated')