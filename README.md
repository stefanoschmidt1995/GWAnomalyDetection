# GWAnomalyDetection

A pipeline designed to detect anomalies within a noisy GW strain time series: each anomaly might correspond to an interesting signal (or to a glitch).

## Our work

How does noise behave? Can we "predict" the noise? Can we distinguish between noise and non-noise? If we understand these issues, we are able to detect some signal buried in noise: it might show up as an anomaly in an otherwise noisy time series.
In order to do so, we need a model that is able to predict future observations of a time series; when the forecast fails, we might be in front of an anomaly. Of course, as the gw signal is tiny, we might need some more sophisticated statistical methods to asses that the failure of the model is due to an anomaly (and not to noise itself).

We are working along two lines:
  * Long-short-term memory (LSTM) networks. A LSTM is a NN specifically designed for forecasting a time series, predicting the future, given the past. If the train the network with a lot of noise from LIGO/Virgo, we hope that the network will be able to do such predictions
  * Autoregressive process based on Maximum Entropy Spectral Analysis. The time series is modelled as and [autoregressive process](https://en.wikipedia.org/wiki/Autoregressive_model), where new observation depend linearly on past observation (plus a gaussian noise term). The method is able to model complex correlation within the data and can be used for forecasting. We use [memspectrum](https://github.com/martini-alessandro/Maximum-Entropy-Spectrum) package for an implementation of the process.
  
Once the two models are accurate enough in predictions, we are not done yet: we need an additional statistics that, given the prediction error of one of the two predictors, computes the probabilty that an anomaly is detected. This part is not trivial and requires careful tuning.

## The repository

This repository keeps to folder:
  * `LSTM`: for everything relevant to LSTM
  * `maxent`: for everything relevant to maximum entropy
  
## TO DOs
A lot of things to do:
  * ...
  * Add here
  
## Authors
+ Melissa López Portilla [m.lopez@uu.nl](mailto:m.lopez@uu.nl)
+ Stefano Schmidt [s.schmidt@uu.nl](mailto:s.schmidt@uu.nl)



