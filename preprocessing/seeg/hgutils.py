def auto_bands(fq_min=4.0749286538265, fq_max=200., scale=7.):
	"""
	Get the frequency bands of interest for the neural signal decomposition.
	Usually these are bands between 4 and 200 Hz, log spaced. 
	These filters were originally chosen by Erik Edwards for his
	thesis work (LH)
	"""
	cts = 2 ** (np.arange(np.log2(fq_min) * scale, np.log2(fq_max) * scale) / scale)
	sds = 10 ** (np.log10(.39) + .5 * (np.log10(cts)))
	return cts, sds

def applyHilbertTransform(X, rate, center, sd):
	"""
	Apply bandpass filtering with Hilbert transform using a Gaussian kernel
	From https://github.com/HamiltonLabUT/MovieTrailers_TIMIT/blob/master/ECoG/preprocessing/applyHilbertTransform.py
	"""
	# frequencies
	T = X.shape[-1]
	freq = fftfreq(T, 1/rate)
	# heaviside kernel
	h = np.zeros(len(freq))
	h[freq > 0] = 2.
	h[0] = 1.
	# bandpass transfer function
	k = np.exp((-(np.abs(freq)-center)**2)/(2*(sd**2)))
	# compute analytical signal
	Xc = ifft(fft(X)*h*k)
	return Xc