# Notes on how to move forward

Two main ideas:

## Forecasting + error measurments
In this framework, we aim to predict the strain and we measure the discrepancy between prediction and true value. An anomaly might show up with a large error.
Some models/interesting papers include:

- [Dynamic Boltzmann machines](https://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/12-Dasgupta-14350.pdf): extension of Boltzmann machine for time-serie forescasting
- [Masked Conditional Neural Networks for Environmental Sound Classification](https://arxiv.org/pdf/1805.10004.pdf): classification of sounds snippets. Not very suitable for streaming data and operates within a supervised approach, but may be interesting.
- Gaussian processes: they are flexible and realiable predictors and are used in time-series analysis. Forecasting is computational expensive and depends on the kernel
- [Conditional Neural Processes](https://arxiv.org/pdf/1807.01613.pdf) (CNP): some sort of mixture between NN and GP.
- [Continuous Time-series Forecasting with Deep and Shallow Stochastic Processes](https://marcpickett.com/cl2018/CL-2018_paper_52.pdf): application of CNP to continuos learning for streaming data. It looks it has very nice predictions


## Feature extraction + clustering
From the raw time-series, we extract a bunch of time dependent features. The features will form a low dimensional space in which we can do some sort of clsutering to detect outliers.
The main question here is: _which features shall represent the GW strain?_
A few options to explore (*please add*):
- [Fractal dimension](https://dcc.ligo.org/DocDB/0177/G2101601/001/fractals_detcharF2F.pdf)
- PSD on the short time scale
- Frequency content of each small bin
- Something related to autocorelation?

More research on literature is needed on this...

## Interesting sofware
- [Librosa](https://librosa.org/doc/latest/tutorial.html): software for audio analysis. It implements some features, maybe useful for streaming data
