## Some notes for GWAnomalyDetection

* Currently, MESA algorithm does fairly good predictions but doesn't get any signal (unless the signal has a very different frequency than the noise)... We might need:

	i) more accurate prediction scheme (LSTM might be better)
	
	ii) More sophisticated analysis of the residuals 

* Clustering of the residuals: the time series of the errors can be reduced in dimensionality and are clustered with DBSCAN (or other unsupervised algorithm). If there is any outlier, we found an anomaly!

* Very interesting NN to predict the sunspots data: see the [website](https://aditya-bhattacharya.net/2020/07/11/time-series-tips-and-tricks/2/). Useful for an inspiration about the LSTM.

* Interesting survey on Outlier Detection for Temporal Data: see [here](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/01/gupta14_tkde.pdf):

* Entropy-based method: when the entropy of a time series increases, this might mean that an decrease in entropy...
This [paper](https://link.springer.com/article/10.1007/s10115-017-1067-8) is about the topic and might serve our purpose: not very interesting (probably beacuse I did not understand it?).

* Dynamic time warping (DTW) is a popular technique for measuring the distance between two time series with temporal deformations: might be interesting to train the LSTM and my precessing mlgw. See on [Wikipedia](https://en.wikipedia.org/wiki/Dynamic_time_warping) for more.

## Signal processing interesting readings:

* Signal processing for dummies by Elena Cuoco (blog): https://getpocket.com/read/3258180325

* Justing's master thesis: https://matheo.uliege.be/bitstream/2268.2/9211/4/Master_thesis_Janquart.pdf

* Factors that impact the PSD (blog): https://sapienlabs.org/factors-that-impact-power-spectrum-density-estimation/

* Understanding the windowing process (blog): https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf

* Digital Signal processing (book): https://doc.lagout.org/science/0_Computer%20Science/9_Others/1_Digital%20Signal%20Processing/The%20Scientist%20and%20Engineer%27s%20Guide%20to%20DSP.pdf

* A guide to LIGO-Virgo detector noise (paper): https://arxiv.org/pdf/1908.11170.pdf

* Empirical mode decomposition (Mathlab code): https://getpocket.com/read/98390696 
