# This script does the following things in order:
# 1. Applies a notch filter at 60,120,180 Hz
# 2. Opens a MNE GUI for the user to manually reject bad channels
# 3. Applies a bipolar average reference to the data
# 4. Opens a MNE GUI for the user to manually reject bad segments
# 5. Saves out the annotated and referenced data as ecog_raw_annotated.fif in the subject's folder

# Local paths, please update according to your machine.
data_path = '/path/to/bids/dataset/'
git_path = '/path/to/git/kurteff2024_code/'

# Imports
import os
import mne
import numpy as np
import re
import matplotlib
matplotlib.rcParams['backend'] = 'TKAgg' # lets you annotate data on plots

# CLI inputs
print("What subject?"); subj = input("> ").replace("sub-","")
print("What block?"); block = input("> ")

# Load raw data
blockid = "_".join([subj,block])
raw_fname = os.path.join(data_path,f"sub-{subj}",blockid,"Raw","ecog_raw.fif")
annot_fname = os.path.join(data_path,f"sub-{subj}",blockid,"Raw","ecog_raw_bipolar_annotated.fif")
raw = mne.io.read_raw_fif(raw_fname,preload=True)

print("Press Enter to power spectral density...")
raw.plot_psd()

input("Press Enter to apply notch filter...")
raw.notch_filter(np.arange(60,120,180))

input("Press Enter to plot raw data. Reject bad channels now.")
raw.plot(scalings='auto')

input("Press Enter to apply a bipolar reference to the data...")
ch_names = raw.info['ch_names']; bipolar_ch_names = []
devices = np.unique([re.sub("\d+","",ch) for ch in ch_names])
for d in devices:
	dchs = [ch for ch in ch_names if d in ch]
	for i,dch in enumerate(dchs):
		if i != len(dchs)-1:
			bipolar_ch_names.append("_".join([dch,dchs[i+1]]))
anodes = list(np.array([bch.split("_") for bch in bipolar_ch_names])[:,0])
cathodes = list(np.array([bch.split("_") for bch in bipolar_ch_names])[:,1])
bipolar_raw = mne.set_bipolar_reference(raw,anodes,cathodes)

input("Press Enter to plot raw data. Annotate bad segments now. (default MNE shortcut for annotation is `a`)")
bipolar_raw.plot(scalings='auto')

input("Press Enter to save data...")
bipolar_raw.save(annot_fname, overwrite=True)