import scipy.io # For .mat files
import h5py # For loading hf5 files
import mne # For loading BrainVision files (EEG)
import numpy as np
from scipy.io import wavfile
import os
import re
import matplotlib.pyplot as plt # For plotting
from matplotlib import cm, rcParams
import random
import itertools as itools
import csv
import logging
import sys
from scipy.signal import spectrogram, resample, hilbert, butter, filtfilt

zs = lambda x: (x-x[np.isnan(x)==False].mean(0))/x[np.isnan(x)==False].std(0)

def ridge(stim, resp, alpha, singcutoff=1e-10, normalpha=False, logger=ridge_logger):
	"""Uses ridge regression to find a linear transformation of [stim] that approximates
	[resp]. The regularization parameter is [alpha].
	Parameters
	----------
	stim : array_like, shape (T, N)
		Stimuli with T time points and N features.
	resp : array_like, shape (T, M)
		Responses with T time points and M separate responses.
	alpha : float or array_like, shape (M,)
		Regularization parameter. Can be given as a single value (which is applied to
		all M responses) or separate values for each response.
	normalpha : boolean
		Whether ridge parameters should be normalized by the largest singular value of stim. Good for
		comparing models with different numbers of parameters.
	Returns
	-------
	wt : array_like, shape (N, M)
		Linear regression weights.
	"""
	try:
		U,S,Vh = np.linalg.svd(stim, full_matrices=False)
	except np.linalg.LinAlgError:
		logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
		from text.regression.svd_dgesvd import svd_dgesvd
		U,S,Vh = svd_dgesvd(stim, full_matrices=False)

	UR = np.dot(U.T, np.nan_to_num(resp))
	#plt.imshow(UR)
	
	# Expand alpha to a collection if it's just a single value
	if isinstance(alpha, float):
		alpha = np.ones(resp.shape[1]) * alpha
	
	# Normalize alpha by the LSV norm
	norm = S[0]
	if normalpha:
		nalphas = alpha * norm
	else:
		nalphas = alpha

	# Compute weights for each alpha
	ualphas = np.unique(nalphas)
	wt = np.zeros((stim.shape[1], resp.shape[1]))
	for ua in ualphas:
		selvox = np.nonzero(nalphas==ua)[0]
		#awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
		awt = np.dot(Vh.T, np.dot(np.diag(S/(S**2+ua**2)), UR[:,selvox]))
		wt[:,selvox] = awt

	return wt

def eigridge(stim, resp, alpha, singcutoff=1e-10, normalpha=False, force_cmode=None, covmat=None, Q=None, L=None, logger=ridge_logger):
	"""Uses ridge regression with eigenvalue decomposition to find a linear transformation of 
	[stim] that approximates [resp]. The regularization parameter is [alpha]. This method seems
	to work better for EEG and ECoG, where the number of time points is larger than the number
	of electrodes.
	Parameters
	----------
	stim : array_like, shape (T, N)
		Stimuli with T time points and N features.
	resp : array_like, shape (T, M)
		Responses with T time points and M separate responses.
	alpha : float or array_like, shape (M,)
		Regularization parameter. Can be given as a single value (which is applied to
		all M responses) or separate values for each response.
	normalpha : boolean
		Whether ridge parameters should be normalized by the largest singular value of stim. Good for
		comparing models with different numbers of parameters.
	Returns
	-------
	wt : array_like, shape (N, M)
		Linear regression weights.
	"""
	if force_cmode is not None:
		cmode = force_cmode
	else:
		cmode = stim.shape[0]<stim.shape[1]

	print("Cmode =",cmode)
	if cmode:
		print("Number of time points is less than the number of features")
	else:
		print("Number of time points is greater than the number of features")

	logger.info("Doing Eigenvalue decomposition on the full stimulus matrix...")

	if cmode: # Make covmat first dim x first dim
		if covmat is None:
			print("stim shape: ",)
			print(stim.shape)
			covmat = np.array(np.dot(stim, stim.T))
		
		print( "Covmat shape: ",)
		print( covmat.shape)
		if Q is None and L is None:
			L, Q = np.linalg.eigh(covmat)
		print( "COV L.T stim.T resp: ",)
		print( stim.T.shape, Q.shape, Q.T.shape, resp.shape)
		Q1 = np.dot(stim.T, Q)
		Q2 = np.dot(Q.T, resp)
	else: # Make covmat second dim x second dim
		if covmat is None: 
			print( "stim shape (not cmode): ", )
			print( stim.shape)
			covmat = np.array(np.dot(stim.T, stim))
	
		print( "Covmat shape: ",)
		print( covmat.shape)
		if Q is None and L is None:
			L, Q = np.linalg.eigh(covmat)

		print( Q.T.shape, stim.T.shape, resp.shape)

		QT_XT_Y = np.dot(Q.T, np.dot(stim.T, resp))

	# Expand alpha to a collection if it's just a single value
	if isinstance(alpha, float):
		alpha = np.ones(resp.shape[1]) * alpha

	# Compute weights for each alpha
	logger.info("Computing weights")
	ualphas = np.unique(alpha)
	wt = np.zeros((stim.shape[1], resp.shape[1]))
	for ua in ualphas:
		selected_elec = np.nonzero(alpha==ua)[0]
		D = np.diag(1 / (L + ua)) # This is for eigridge
		if cmode:
			#awt = reduce(np.dot( [Q1, D, Q2[:,selected_elec]]))
			awt = np.dot( Q1, np.dot(D, Q2[:,selected_elec]))
		else:
			awt = np.dot(Q, np.dot(D, QT_XT_Y[:,selected_elec]))
		wt[:,selected_elec] = awt

	return wt

def bootstrap_ridge(Rstim, Rresp, Pstim, Presp, alphas, nboots, chunklen, nchunks,
					corrmin=0.2, joined=None, singcutoff=1e-10, normalpha=False, single_alpha=False,
					use_corr=True, logger=ridge_logger, return_wts=True, use_svd=False):
	"""Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.
	[nchunks] random chunks of length [chunklen] will be taken from [Rstim] and [Rresp] for each regression
	run.  [nboots] total regression runs will be performed.  The best alpha value for each response will be
	averaged across the bootstraps to estimate the best alpha for that response.
	
	If [joined] is given, it should be a list of lists where the STRFs for all the electrodes/voxels in each sublist 
	will be given the same regularization parameter (the one that is the best on average).
	
	Parameters
	----------
	Rstim : array_like, shape (TR, N)
		Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
	Rresp : array_like, shape (TR, M)
		Training responses with TR time points and M different responses (electrodes, voxels, neurons, what-have-you).
		Each response should be Z-scored across time.
	Pstim : array_like, shape (TP, N)
		Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
	Presp : array_like, shape (TP, M)
		Test responses with TP time points and M different responses. Each response should be Z-scored across
		time.
	alphas : list or array_like, shape (A,)
		Ridge parameters that will be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
	nboots : int
		The number of bootstrap samples to run. 15 to 30 works well.
	chunklen : int
		On each sample, the training data is broken into chunks of this length. This should be a few times 
		longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of length 10.
	nchunks : int
		The number of training chunks held out to test ridge parameters for each bootstrap sample. The product
		of nchunks and chunklen is the total number of training samples held out for each sample, and this 
		product should be about 20 percent of the total length of the training data.
	corrmin : float in [0..1]
		Purely for display purposes. After each alpha is tested for each bootstrap sample, the number of 
		responses with correlation greater than this value will be printed. For long-running regressions this
		can give a rough sense of how well the model works before it's done.
	joined : None or list of array_like indices
		If you want the STRFs for two (or more) responses to be directly comparable, you need to ensure that
		the regularization parameter that they use is the same. To do that, supply a list of the response sets
		that should use the same ridge parameter here. For example, if you have four responses, joined could
		be [np.array([0,1]), np.array([2,3])], in which case responses 0 and 1 will use the same ridge parameter
		(which will be parameter that is best on average for those two), and likewise for responses 2 and 3.
	singcutoff : float
		The first step in ridge regression is computing the singular value decomposition (SVD) of the
		stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
		to zero and the corresponding singular vectors will be noise. These singular values/vectors
		should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
		singular values less than singcutoff will be removed.
	normalpha : boolean
		Whether ridge parameters (alphas) should be normalized by the largest singular value (LSV)
		norm of Rstim. Good for rigorously comparing models with different numbers of parameters.
	single_alpha : boolean
		Whether to use a single alpha for all responses. Good for identification/decoding.
	use_corr : boolean
		If True, this function will use correlation as its metric of model fit. If False, this function
		will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
		this can make a big difference -- highly regularized solutions will have very small norms and
		will thus explain very little variance while still leading to high correlations, as correlation
		is scale-free while R**2 is not.
	
	Returns
	-------
	wt : array_like, shape (N, M)
		Regression weights for N features and M responses.
	corrs : array_like, shape (M,)
		Validation set correlations. Predicted responses for the validation set are obtained using the regression
		weights: pred = np.dot(Pstim, wt), and then the correlation between each predicted response and each 
		column in Presp is found.
	alphas : array_like, shape (M,)
		The regularization coefficient (alpha) selected for each electrode/voxel using bootstrap cross-validation.
	bootstrap_corrs : array_like, shape (A, M, B)
		Correlation between predicted and actual responses on randomly held out portions of the training set,
		for each of A alphas, M electrodes, and B bootstrap samples.
	valinds : array_like, shape (TH, B)
		The indices of the training data that were used as "validation" for each bootstrap sample.
	"""
	nresp, nvox = Rresp.shape
	valinds = [] # Will hold the indices into the validation data for each bootstrap
	
	Rcmats = []
	for bi in counter(range(nboots), countevery=1, total=nboots):
		logger.info("Selecting held-out test set..")
		allinds = range(nresp)
		indchunks = list(zip(*[iter(allinds)]*chunklen))
		random.shuffle(indchunks)
		heldinds = list(itools.chain(*indchunks[:nchunks]))
		notheldinds = list(set(allinds)-set(heldinds))
		valinds.append(heldinds)
		
		RRstim = Rstim[notheldinds,:]
		PRstim = Rstim[heldinds,:]
		RRresp = Rresp[notheldinds,:]
		PRresp = Rresp[heldinds,:]
		
		if use_svd:
			# Run ridge regression using this test set
			Rcmat = ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
						   corrmin=corrmin, singcutoff=singcutoff,
						   normalpha=normalpha, use_corr=use_corr,
						   logger=logger)
		else:
			# Run ridge regression using this test set
			Rcmat = eigridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
						   corrmin=corrmin, singcutoff=singcutoff,
						   normalpha=normalpha, use_corr=use_corr,
						   logger=logger)
		
		Rcmats.append(Rcmat)
	
	# Find best alphas
	if nboots>0:
		allRcorrs = np.dstack(Rcmats)
	else:
		allRcorrs = None
	
	if not single_alpha:
		if nboots==0:
			raise ValueError("You must run at least one cross-validation step to assign "
							 "different alphas to each response.")
		
		logger.info("Finding best alpha for each electrode..")
		if joined is None:
			# Find best alpha for each electrode
			meanbootcorrs = allRcorrs.mean(2)
			bestalphainds = np.argmax(meanbootcorrs, 0)
			valphas = alphas[bestalphainds]
		else:
			# Find best alpha for each group of electrode
			valphas = np.zeros((nvox,))
			for jl in joined:
				# Mean across electrodes in the set, then mean across bootstraps
				jcorrs = allRcorrs[:,jl,:].mean(1).mean(1)
				bestalpha = np.argmax(jcorrs)
				valphas[jl] = alphas[bestalpha]
	else:
		logger.info("Finding single best alpha..")
		if nboots==0:
			if len(alphas)==1:
				bestalphaind = 0
				bestalpha = alphas[0]
			else:
				raise ValueError("You must run at least one cross-validation step "
								 "to choose best overall alpha, or only supply one"
								 "possible alpha value.")
		else:
			meanbootcorr = allRcorrs.mean(2).mean(1)
			bestalphaind = np.argmax(meanbootcorr)
			bestalpha = alphas[bestalphaind]
		
		valphas = np.array([bestalpha]*nvox)
		logger.info("Best alpha = %0.3f"%bestalpha)

	if return_wts:    
		# Find weights
		logger.info("Computing weights for each response using entire training set..")
		if use_svd:
			wt = ridge(Rstim, Rresp, valphas, singcutoff=singcutoff, normalpha=normalpha)
		else:
			wt = eigridge(Rstim, Rresp, valphas, singcutoff=singcutoff, normalpha=normalpha)

		# Predict responses on prediction set
		logger.info("Predicting responses for predictions set..")

		if wt.shape[0]==Pstim.shape[1]+1:
			logger.info("Using intercept in prediction")
			pred = np.dot(Pstim, wt[1:]) + wt[0]
		else:
			pred = np.dot(Pstim, wt)

		# Find prediction correlations
		nnpred = np.nan_to_num(pred)
		if use_corr:
			corrs = np.nan_to_num(np.array([np.corrcoef(Presp[:,ii], nnpred[:,ii].ravel())[0,1]
											for ii in range(Presp.shape[1])]))
		else:
			resvar = (Presp-pred).var(0)
			Rsqs = 1 - (resvar / Presp.var(0))
			corrs = np.sqrt(np.abs(Rsqs)) * np.sign(Rsqs)

		return wt, corrs, valphas, allRcorrs, valinds, pred, Pstim ## LH ADDED
	else:
		return valphas, allRcorrs, valinds

def ridge_corr(Rstim, Pstim, Rresp, Presp, alphas, normalpha=False, corrmin=0.2,
               singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. The regression weights are NOT returned, because
    computing the correlations without computing regression weights is much, MUCH faster.
    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (electrodes, voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for each alpha.
    
    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    ## Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: %0.03f"%norm)
    if normalpha:
        nalphas = alphas * norm
    else:
        nalphas = alphas

    ## Precompute some products for speed
    UR = np.dot(U.T, Rresp) ## Precompute this matrix product for speed
    PVh = np.dot(Pstim, Vh.T) ## Precompute this matrix product for speed
    
    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(Presp)
    #Prespvar = Presp.var(0)
    Prespvar_actual = Presp.var(0)
    Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (Prespvar_actual - Prespvar).mean())
    Rcorrs = [] ## Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        #D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter 
        D = S / (S ** 2 + na ** 2) ## Reweight singular vectors by the (normalized?) ridge parameter
        
        pred = np.dot(mult_diag(D, PVh, left=False), UR) ## Best (1.75 seconds to prediction in test)
        # pred = np.dot(mult_diag(D, np.dot(Pstim, Vh.T), left=False), UR) ## Better (2.0 seconds to prediction in test)
        
        # pvhd = reduce(np.dot, [Pstim, Vh.T, D]) ## Pretty good (2.4 seconds to prediction in test)
        # pred = np.dot(pvhd, UR)
        
        # wt = reduce(np.dot, [Vh.T, D, UR]).astype(dtype) ## Bad (14.2 seconds to prediction in test)
        # wt = reduce(np.dot, [Vh.T, D, U.T, Rresp]).astype(dtype) ## Worst
        # pred = np.dot(Pstim, wt) ## Predict test responses

        if use_corr:
            #prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
            #Rcorr = np.array([np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1] for ii in range(Presp.shape[1])]) ## Slowly compute correlations
            #Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()/(prednorms*Prespnorms) ## Efficiently compute correlations
            Rcorr = (zPresp * zs(pred)).mean(0)
        else:
            ## Compute variance explained
            resvar = (Presp - pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            Rcorr = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)
            
        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
        
        log_template = "Training: alpha=%0.3f, mean corr=%0.5f, max corr=%0.5f, over-under(%0.2f)=%d"
        log_msg = log_template % (a,
                                  np.mean(Rcorr),
                                  np.max(Rcorr),
                                  corrmin,
                                  (Rcorr>corrmin).sum()-(-Rcorr>corrmin).sum())
        logger.info(log_msg)
    
    return Rcorrs

def eigridge_corr(Rstim, Pstim, Rresp, Presp, alphas, normalpha=False, corrmin=0.2,
               singcutoff=1e-10, use_corr=True, force_cmode=None, covmat=None, logger=ridge_logger):
    """Uses ridge regression with eigenvalue decomposition (instead of SVD)
    to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. The regression weights are NOT returned, because
    computing the correlations without computing regression weights is much, MUCH faster.
    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (electrodes, voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for each alpha.
    
    """
    if force_cmode is not None:
        cmode = force_cmode
    else:
        cmode = Rstim.shape[0]<Rstim.shape[1]

    # print( "Cmode =",cmode)
    # if cmode:
    #     print( "Number of time points is less than the number of features")
    # else:
    #     print( "Number of time points is greater than the number of features")

    logger.info("Doing Eigenvalue decomposition...")

    if cmode: # Make covmat first dim x first dim
        if covmat is None:
            # print( "Rstim shape: ",)
            # print( Rstim.shape)
            covmat = np.array(np.dot(Rstim, Rstim.T))
        
        print( "Covmat shape: ",)
        print( covmat.shape)
        L, Q = np.linalg.eigh(covmat)
        print( "COV L.T Rstim.T Rresp: ",)
        print( Rstim.T.shape, Q.shape, Q.T.shape, Rresp.shape)
        Q1 = np.dot(Rstim.T, Q)
        Q2 = np.dot(Q.T, Rresp)
    else: # Make covmat second dim x second dim
        if covmat is None: 
            # print( "Rstim shape (not cmode): ", )
            # print( Rstim.shape)
            covmat = np.array(np.dot(Rstim.T, Rstim))
    
        # print( "Covmat shape: ",)
        # print( covmat.shape)
        L, Q = np.linalg.eigh(covmat)

        # print( Q.T.shape, Rstim.T.shape, Rresp.shape)

        QT_XT_Y = np.dot(Q.T, np.dot(Rstim.T, Rresp))

    # USV^T, mat = Q*L*Q.T

    ## Precompute some products for speed
    XQ = np.dot(Pstim, Q) ## Precompute this matrix product for speed

    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(Presp)
    Prespvar = Presp.var(0)
    #Prespvar_actual = Presp.var(0)
    #Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    #logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (Prespvar_actual - Prespvar).mean())
    Rcorrs = [] ## Holds training correlations for each alpha
    for a in alphas:
        D = 1 / (L + a) # This is for eigridge

        # if cmode:
        #     pred = np.dot(PStim, reduce(np.dot([Q1, D, Q2])))
        # else:
        #     pred = np.dot(PStim, reduce(np.dot([Q, D, QT_XT_Y])))

        pred = np.dot(mult_diag(D, XQ, left=False), QT_XT_Y) ## Best (1.75 seconds to prediction in test)
        # pred = np.dot(mult_diag(D, np.dot(Pstim, Vh.T), left=False), UR) ## Better (2.0 seconds to prediction in test)
        
        # pvhd = reduce(np.dot, [Pstim, Vh.T, D]) ## Pretty good (2.4 seconds to prediction in test)
        # pred = np.dot(pvhd, UR)
        
        # wt = reduce(np.dot, [Vh.T, D, UR]).astype(dtype) ## Bad (14.2 seconds to prediction in test)
        # wt = reduce(np.dot, [Vh.T, D, U.T, Rresp]).astype(dtype) ## Worst
        # pred = np.dot(Pstim, wt) ## Predict test responses

        if use_corr:
            #prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
            #Rcorr = np.array([np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1] for ii in range(Presp.shape[1])]) ## Slowly compute correlations
            #Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()/(prednorms*Prespnorms) ## Efficiently compute correlations
            Rcorr = (zPresp * zs(pred)).mean(0)
        else:
            ## Compute variance explained
            resvar = (Presp - pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            Rcorr = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)
            
        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
        
        log_template = "Training: alpha=%0.3f, mean corr=%0.5f, max corr=%0.5f, over-under(%0.2f)=%d"
        log_msg = log_template % (a,
                                  np.mean(Rcorr),
                                  np.max(Rcorr),
                                  corrmin,
                                  (Rcorr>corrmin).sum()-(-Rcorr>corrmin).sum())
        logger.info(log_msg)
    
    return Rcorrs

def make_delayed(stim, delays, circpad=False):
	"""Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
	(in samples).
	
	If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
	"""
	nt,ndim = stim.shape
	dstims = []
	for di,d in enumerate(delays):
		dstim = np.zeros((nt, ndim))
		if d<0: ## negative delay
			dstim[:d,:] = stim[-d:,:]
			if circpad:
				dstim[d:,:] = stim[:-d,:]
		elif d>0:
			dstim[d:,:] = stim[:-d,:]
			if circpad:
				dstim[:d,:] = stim[-d:,:]
		else: ## d==0
			dstim = stim.copy()
		dstims.append(dstim)
	return np.hstack(dstims)

def get_feats(model_number='model1',mode='eeg',return_dict=False,extend_labels=False):
	'''
	onsetProd helper function. Returns a list of features given model number.
	'''
	if extend_labels == False:
		task_labels = ['spkr','mic','el','sh']
	else:
		task_labels = ['perception','production','predictable','unpredictable']
	if model_number in ['model8', 'model9','model10', 'model11']:
		if model_number in ['model8', 'model10']: # Manner only
			features_dict = {
				'plosive': ['p','pcl','t','tcl','k','kcl','b','bcl','d','dcl','g','gcl','q'],
				'fricative': ['f','v','th','dh','s','sh','z','zh','hh','hv','ch','jh'],
				'syllabic': ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux'],
				'nasal': ['m','em','n','en','ng','eng','nx'],
				'voiced':   ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','dh','z','v','b','bcl','d','dcl','g','gcl','m','em','n','en','eng','ng','nx','q','jh','zh'],
				'obstruent': ['b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx','f', 'g', 'gcl', 'hh', 'hv','jh', 'k', 'kcl', 'p', 'pcl', 'q', 's', 'sh','t', 'tcl', 'th','v','z', 'zh','q'],
				'sonorant': ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','m', 'n', 'ng', 'eng', 'nx','en','em'],
			}
		if model_number in ['model9', 'model11']: # Place only
			features_dict = {
				'dorsal': ['y','w','k','kcl', 'g','gcl','eng','ng'],
				'coronal': ['ch','jh','sh','zh','s','z','t','tcl','d','dcl','n','th','dh','l','r'],
				'labial': ['f','v','p','pcl','b','bcl','m','em','w'],
				'high': ['uh','ux','uw','iy','ih','ix','ey','eh','oy'],
				'front': ['iy','ih','ix','ey','eh','ae','ay'],
				'low': ['aa','ao','ah','ax','ae','aw','ay','axr','ow','oy'],
				'back': ['aa','ao','ow','ah','ax','ax-h','uh','ux','uw','axr','aw'],
				'voiced':   ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','dh','z','v','b','bcl','d','dcl','g','gcl','m','em','n','en','eng','ng','nx','q','jh','zh'],
			}
	else:
		features_dict = {
			'dorsal': ['y','w','k','kcl', 'g','gcl','eng','ng'],
			'coronal': ['ch','jh','sh','zh','s','z','t','tcl','d','dcl','n','th','dh','l','r'],
			'labial': ['f','v','p','pcl','b','bcl','m','em','w'],
			'high': ['uh','ux','uw','iy','ih','ix','ey','eh','oy'],
			'front': ['iy','ih','ix','ey','eh','ae','ay'],
			'low': ['aa','ao','ah','ax','ae','aw','ay','axr','ow','oy'],
			'back': ['aa','ao','ow','ah','ax','ax-h','uh','ux','uw','axr','aw'],
			'plosive': ['p','pcl','t','tcl','k','kcl','b','bcl','d','dcl','g','gcl','q'],
			'fricative': ['f','v','th','dh','s','sh','z','zh','hh','hv','ch','jh'],
			'syllabic': ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux'],
			'nasal': ['m','em','n','en','ng','eng','nx'],
			'voiced':   ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','dh','z','v','b','bcl','d','dcl','g','gcl','m','em','n','en','eng','ng','nx','q','jh','zh'],
			'obstruent': ['b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx','f', 'g', 'gcl', 'hh', 'hv','jh', 'k', 'kcl', 'p', 'pcl', 'q', 's', 'sh','t', 'tcl', 'th','v','z', 'zh','q'],
			'sonorant': ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay','eh','ey','ih', 'ix', 'iy','ow', 'oy','uh', 'uw', 'ux','w','y','el','l','r','m', 'n', 'ng', 'eng', 'nx','en','em'],
		}
	features = [f for f in features_dict.keys()]
	if mode == 'ecog':
		emg = False
	elif model_number[-1] == 'e':
		emg = False
		model_number = model_number[:-1]
	else:
		emg = True
	if model_number in ['model1', 'model8', 'model9']:
		y_labels = features + task_labels
	elif model_number in ['model2', 'model10', 'model11']:
		y_labels = features + [f'spkr_{s}' for s in features] + [f'mic_{s}' for s in features] + ['spkr','mic','el','sh']
	elif model_number == 'model3':
		y_labels = features + task_labels[:2]
	elif model_number == 'model4':
		y_labels = features + task_labels[2:]
	elif model_number == 'model5':
		y_labels = features + task_labels + ['spkr_onset','mic_onset']
	elif model_number == 'model6':
		y_labels = y_labels = features + [f'spkr_{s}' for s in features] + [f'mic_{s}' for s in features] + ['spkr','mic','el','sh','spkr_onset','mic_onset']
	elif model_number == 'model7':
		y_labels = features
	elif model_number == 'model12':
		y_labels = task_labels
	elif model_number == 'model13':
		y_labels = features
	if emg == True:
		y_labels.append('emg')
	if return_dict == False:
		return y_labels
	else:
		return features_dict

def load_model_inputs(model_input_h5_fpath,model_number,mode='eeg'):
	'''
	Loads model from hdf5 and indexes it accordingly for a given model number.
	'''
	all_features = get_feats('model2')
	all_phnfeat = [all_features.index(f) for f in all_features[:-5] if "spkr_" not in f and "mic_" not in f]
	spkr_phnfeat = [all_features.index(f) for f in all_features[:-5] if "spkr_" in f]
	mic_phnfeat = [all_features.index(f) for f in all_features[:-5] if "mic_" in f]
	spkr_mic_task_feats = [all_features.index(f) for f in all_features[-5:] if f in ['spkr','mic']]
	el_sh_task_feats = [all_features.index(f) for f in all_features[-5:] if f in ['el','sh']]
	emg_task_feat = [all_features.index('emg')]
	# Load data
	if os.path.isfile(model_input_h5_fpath):
		with h5py.File(model_input_h5_fpath,'r') as f:
			tStim = np.array(f.get('tStim'))
			tResp = np.array(f.get('tResp'))
			vStim = np.array(f.get('vStim'))
			vResp = np.array(f.get('vResp'))
	else:
		raise Exception(f"File does not exist: {model_input_h5_fpath}")
	# Index specific features according to model number
	if 'model1' in model_number:
		# 18(+1) features: 14 phnfeat + 4 task (+ emg)
		feat_idxs = all_phnfeat + spkr_mic_task_feats + el_sh_task_feats
	if 'model2' in model_number:
		# 46(+1) features: 14*3 phnfeat + 4 task (+ emg)
		feat_idxs = all_phnfeat + spkr_phnfeat + mic_phnfeat + spkr_mic_task_feats + el_sh_task_feats
	if 'model3' in model_number:
		# 16(+1) features: 14 phnfeat + 2 task (+ emg)
		feat_idxs = all_phnfeat + spkr_mic_task_feats
	if 'model4' in model_number:
		# 16(+1) features: 14 phnfeat +2 task (+ emg)
		feat_idxs = all_phnfeat + el_sh_task_feats
	if mode == 'eeg' and model_number[-1] != 'e':
		# Add the EMG (if the model includes it)
		feat_idxs = feat_idxs + emg_task_feat
	tStim = tStim[:,feat_idxs]
	vStim = vStim[:,feat_idxs]
	return tStim, tResp, vStim, vResp

def mtrf(resp, stim,
	delay_min=0, delay_max=0.6, wt_pad=0.0, alphas=np.hstack((0, np.logspace(-3,5,20))),
	use_corr=True, single_alpha=True, nboots=20, sfreq=128, vResp=[],vStim=[], flip_resp=False,return_pred=False):
	'''
	Run the mTRF model.
	* wt_pad: Amount of padding for delays, since edge artifacts can make weights look weird
	* use_corr: Use correlation between predicted and validation set as metric for goodness of fit
	* single_alpha: Use the same alpha value for all electrodes (helps with comparing across sensors)
	* Logspace was previously -1 to 7, changed for smol stim strf in may 20
	'''
	# Populate stim and resp lists (these will be used to get tStim and tResp, or vStim and vResp)
	stim_list = []
	stim_sum= []
	train_or_val = [] # Mark for training or for validation set
	np.random.seed(6655321)
	if flip_resp == True:
		resp = resp.T
		if len(vResp) >= 1:
			vResp = vResp.T
	# Load stimulus and response
	if resp.shape[1] != stim.shape[0]:
		logging.warning("Resp and stim do not match! This is a problem")
	nchans, ntimes = resp.shape
	print(nchans, ntimes)
	# RUN THE STRFS
	# For logging compute times, debug messages
	logging.basicConfig(level=logging.DEBUG)
	delays = np.arange(np.floor((delay_min-wt_pad)*sfreq), np.ceil((delay_max+wt_pad)*sfreq), dtype=int) 
	# print("Delays:", delays)
	# Regularization parameters (alphas - also sometimes called lambda)
	# MIGHT HAVE TO CHANGE ALPHAS RANGE... e.g. alphas = np.hstack((0, np.logspace(-2,5,20)))
	# alphas = np.hstack((0, np.logspace(2,8,20))) # Gives you 20 values between 10^2 and 10^8
	# alphas = np.hstack((0, np.logspace(-1,7,20))) # Gives you 20 values between 10^-2 and 10^5
	# alphas = np.logspace(1,8,20) # Gives you 20 values between 10^1 and 10^8
	nalphas = len(alphas)
	all_wts = []
	all_corrs = []
	# Train on 80% of the trials, test on 
	# the remaining 20%.
	# Z-scoring function (assumes time is the 0th dimension)
	resp = zs(resp.T).T
	if len(vResp) >= 1:
		vResp = zs(vResp.T).T
	if len(vResp) == 0 and len(vStim) == 0:
		print("Creating vResp and vStim using an automated 80-20 split...")
		# Create training and validation response matrices.
		# Time must be the 0th dimension.
		tResp = resp[:,:int(0.8*ntimes)].T
		vResp = resp[:,int(0.8*ntimes):].T
		# Create training and validation stimulus matrices
		tStim_temp = stim[:int(0.8*ntimes),:]
		vStim_temp = stim[int(0.8*ntimes):,:]

	else: # if vResp and vStim were passed into the function
		print("Using training/validation split passed into the func...")
		tResp = resp.T
		vResp = vResp.T
		tStim_temp = stim
		vStim_temp = vStim
	tStim = make_delayed(tStim_temp, delays)
	vStim = make_delayed(vStim_temp, delays)
	chunklen = int(len(delays)*4) # We will randomize the data in chunks 
	nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype(int)
	nchans = tResp.shape[1] # Number of electrodes/sensors

	# get a strf
	print(tStim.shape, vStim.shape)
	print(tResp.shape, vResp.shape)
	wt, corrs, valphas, allRcorrs, valinds, pred, Pstim = ridge.bootstrap_ridge(tStim, tResp, vStim, vResp, 
																		  alphas, nboots, chunklen, nchunks, 
																		  use_corr=use_corr,  single_alpha = single_alpha, 
																		  use_svd=False, corrmin = 0.05,
																		  joined=[np.array(np.arange(nchans))])
	print("wt shape:")
	print(wt.shape)
	
	# If we decide to add some padding to our model to account for edge artifacts, 
	# get rid of it before returning the final strf
	if wt_pad>0:
		print("Reshaping weight matrix to get rid of padding on either side")
		orig_delays = np.arange(np.floor(delay_min*sfreq), np.ceil(delay_max*sfreq), dtype=np.int) 

		# This will be a boolean mask of only the "good" delays (not part of the extra padding)
		good_delays = np.zeros((len(delays), 1), dtype=np.bool)
		int1, int2, good_inds = np.intersect1d(orig_delays,delays,return_indices=True)
		for g in good_inds:
			good_delays[g] = True	#wt2 = wt.reshape((len(delays), -1, wt.shape[1]))[len(np.where(delays<0)[0]):-(len(np.where(delays<0)[0])),:,:]
		print(delays)
		print(orig_delays)
		# Reshape the wt matrix so it is now only including the original delay_min to delay_max time period instead
		# of delay_min-wt_pad to delay_max+wt_pad
		wt2 = wt.reshape((len(delays), -1, wt.shape[1])) # Now this will be ndelays x nfeat x nchans
		wt2 = wt2[good_delays.ravel(), :, :].reshape(-1, wt2.shape[2])
	else:
		wt2 = wt
	print(wt2.shape)
	all_wts.append(wt2)
	all_corrs.append(corrs)
	if return_pred:
		return(all_corrs,all_wts,tStim,tResp,vStim,vResp,valphas,pred)
	else:
		return(all_corrs, all_wts, tStim, tResp, vStim, vResp, valphas)	