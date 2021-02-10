#This file is for running the pipeline and saving the whitened data on a text file
#Some options can be set: see https://lscsoft.docs.ligo.org/gstlal/gstlal/bin/gstlal_play.html
#Or type gstlal_play -h

export GSTLAL_FIR_WHITEN=0
gstlal_play --data-source frames \
	--frame-cache frame.cache \
	--channel-name V1=Hrec_hoft_16384Hz \
	--amplification 1 \
	--gps-start-time 1240213455 \
	--gps-end-time 1240213555 \
        --high-pass-filter 40 --low-pass-filter 1000 \
	--whiten \
	--output test.wav
