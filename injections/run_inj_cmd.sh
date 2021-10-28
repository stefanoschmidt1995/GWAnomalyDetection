M_MIN=10
M_MAX=15
Q_MIN=1.0
Q_MAX=5.0
S1_MAX=0.

OUT_FILE=./prec_inj.xml

lalapps_inspinj \
                --m-distr totalMassRatio \
                --gps-start-time 1126250415 \
                --gps-end-time   1126260415 \
                --enable-spin \
                --min-spin1 0.0 \
                --max-spin1 $S1_MAX \
                --min-kappa1 -1.0 \
                --max-kappa1 1.0 \
                --min-spin2 0. \
                --max-spin2 0. \
                --min-mratio $Q_MIN \
                --max-mratio $Q_MAX \
                --min-mtotal $M_MIN \
                --max-mtotal $M_MAX \
                --dchirp-distr uniform \
                --min-distance 10000 \
                --max-distance 250000 \
                --waveform IMRPhenomPv2 \
                --f-lower 10 \
                --i-distr uniform \
                --coa-phase-distr uniform \
                --l-distr random \
                --t-distr uniform \
                --time-step 1 \
                --time-interval 0 \
                --taper-injection startend \
                --seed 0 \
                --output $OUT_FILE

ligolw_no_ilwdchar $OUT_FILE

#To display the number of injections:
echo Performed $(ligolw_print -t sim_inspiral -c mass1 $OUT_FILE | wc -l) injections
