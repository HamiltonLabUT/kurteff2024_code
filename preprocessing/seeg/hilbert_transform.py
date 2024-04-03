# Make sure you have run either annotate_raw_data.py or bipolar_reference first!

# The script does the following things in order:

# Local paths, please update according to your machine.
data_path = '/path/to/bids/dataset/'
git_path = '/path/to/git/kurteff2024_code/'

# Change this variable based on which type of annotated data you would like to Hilbert transform.
data_type = 'car' # car or bipolar are the acceptable options
if data_type not in ['car', 'bipolar']:
	raise Exception("Please specify data type as either CAR or bipolar referencing.")
fname_ext = "ecog_raw.fif" if data_type == "car" else "ecog_raw_bipolar_annotated.fif"

# Imports
import os
import mne
import numpy as np
from tqdm import tqdm
from . import hgutils

# CLI inputs
print("What subject?"); subj = input("> ").replace("sub-","")
print("What block?"); block = input("> ")

# Load annotated data
blockid = "_".join([subj,block])
raw_fname = os.path.join(data_path,f"sub-{subj}",blockid,"Raw",fname_ext)
raw = mne.io.read_raw_fif(raw_fname,preload=True)
fs = raw.info['sfreq']

nchans = len([x['kind'] for x in raw.info['chs'] if (x['kind']==902 or x['kind']==2)]) # eeg or ecog
nstimchans = len([x['kind'] for x in raw.info['chs'] if x['kind']==3])
ch_types = (['ecog'] * nchans) + (['stim'] * nstimchans)
f_low, f_high = 70, 150 # the size of our high gamma band
cts, sds = hgutils.auto_bands()
sds = sds[(cts>=f_low) & (cts<=f_high)]; cts = cts[(cts>=f_low) & (cts<=f_high)]

# Run Hilbert transform
dat = []
for i, (ct, sd) in enumerate(tqdm(zip(cts, sds), 'applying Hilbert transform...', total=len(cts))):
	hilbdat = applyHilbertTransform(raw.get_data()[:nchans,:], raw_sf, ct, sd)
	dat.append(np.abs(hilbdat.real.astype('float32') + 1j*hilbdat.imag.astype('float32')))
hilbmat = np.array(np.hstack((dat))).reshape(dat[0].shape[0], -1, dat[0].shape[1])
hg_signal = hilbmat.mean(1) # Get the average across the relevant bands 
rmax = raw.get_data()[:nchans,:].max()
hg_signal = hg_signal/hg_signal.max()*rmax # Rescale to maximum of raw so plot works

# Create high-gamma raw instance
hg_signal_nobad = hg_signal.copy()
for i, a in enumerate(raw.annotations.onset):
	onset_samp = np.int((a*raw_sf)-raw.first_samp)
	offset_samp = onset_samp + np.int(raw.annotations.duration[i]*hg_fs)
	hg_signal_nobad[:,onset_samp] = np.nan
	hg_signal_nobad[:,offset_samp] = np.nan
hg_signal = (hg_signal - np.expand_dims(np.nanmean(hg_signal_nobad, axis=1), axis=1) )/np.expand_dims(np.nanstd(hg_signal_nobad, axis=1), axis=1)
audio_ds = raw.copy().get_data()[nchans:,:]
hg_all = np.vstack((hg_signal, audio_ds))
hg_info = mne.create_info(raw.info['ch_names'], raw_sf, ch_types)
hgdat = mne.io.RawArray(hg_all, hg_info)
if raw.annotations: # if we rejected something reject it in HG also
	for annotation in raw.annotations:
		# Add annotations from raw to hg data
		onset = (annotation['onset']-(raw.first_samp/512)) # convert start time for clinical data   
		duration = annotation['duration']
		description = annotation['description']
		hgdat.annotations.append(onset,duration,description)
hg_fs = 100; hgdat.resample(hg_fs) # Resample to 100 Hz

input("Press Enter to plot high-gamma band signal and reject bad segments...")
hgdat.plot(scalings='auto')

input("Press Enter to save Hilbert-transformed data...")
hgfile = os.path.join(data_path, f"sub-{subj}", blockid, f"HilbAA_{f_low}to{f_high}_8band", f"ecog_hilbAA{f_low}to{f_high}.fif")
hgdat.save(hgfile, overwrite=True)
