## kurteff2024
This is a repo containing the necessary code to reproduce every analysis, result, and figure in Kurteff et al., "Processing of auditory feedback in perisylvian and insular cortex." The paper is available as a preprint on [bioRxiv](http://dx.doi.org/10.1101/2024.05.14.593257) and is published in [J Neurosci](http://dx.doi.org/10.1523/JNEUROSCI.1109-24.2024).

Most of the code is Python-based, with a few exceptions:

* Mixed-effects models were fit in R
* Some of our imaging pipeline uses MATLAB
* The task used during data collection was coded in Swift

### Subfolder explanation

To fully recreate all data, I would suggest running through the folders in this order:

* `preprocessing` contains code for extracting neural time series and imaging from the raw data.
    * `preprocessing/seeg` contains code for preprocessing raw sEEG data into Hilbert-transformed high gamma analytic amplitude timeseries used in our analyses.
    * `preprocessing/events` contains code for extracting the sentence and phone timing from the audio and creating event files for use in MNE-python.
    * `preprocessing/imaging` contains code for stereotactic electrode localization and warping, as well as for generating 3D reconstructions from patients' MRIs.
        * `preprocessing/imaging/img_pipe` is a local fork of https://github.com/ChangLabUcsf/img_pipe.
* `analysis` contains code for conducting the primary analyses of the task.
    * `analysis/mtrf` contains the code for the multivariate temporal receptive field analysis. hdf5 files are not included to save storage, but will be populated upon running the scripts.
    * `analysis/cnmf` contains the code for the convex non-negative matrix factorization analysis. hdf5 files are not included to save storage, but will be populated upon running the scripts.
    * `analysis/speechmotor` contains the code for analyzing the speech motor control task.
* `figures` contains code for reproducing each panel of each figure. PDFs are not included to save storage, but will be populated upon running the scripts. Prerequisites for each panel are listed in the preamble of each Jupyter notebook.
* `stats` contains code for reproducing the statistical analyses.
    * `stats/bootstraps_and_permutations/` contains code for calculating statistical significance through bootstrap and permutation _t_-tests.
    * `stats/lme` contains code for generating the necessary csv files for, and running, linear mixed-effects models.
* `task` isn't necessary to run if you wish to replicate the results using the OpenNeuro dataset, but rather contains the code used to create the task administered to the participants during data collection.
    * `task/mocha` contains the base task which uses stimuli from the MOCHA-TIMIT corpus (Wrench 1989).
    * `task/easy_reading` contains a modified version with less complex sentences but the same phonemic distribution as the base task.
    * `task/speechmotor` contains the code for the speech motor control task run in a subset of participants.

### Requirements

* Python 3.9+
* MATLAB R2021b+
* R 4.2+
* XCode 14.3.1

#### List of third-party Python packages used
Non-pip-installable software will be imported from local scripts.

* fuzzywuzzy 0.18.0
* h5py 3.3.0
* librosa 0.9.2
* matplotlib 3.4.3
* mayavi 4.7.2
* mne 1.1.1
* nibabel 3.2.1
* nilearn 0.8.1
* nipy 0.5.0
* numpy 1.21.6
* pandas 1.3.3
* praatio 4.4.0
* pybv 0.7.4
* pyFFTW 0.12.0
* scipy 1.11.2
* seaborn 0.11.2
* tqdm 4.62.3
