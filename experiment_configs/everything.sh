# these are already the defaults but written explicitly
export N_LAYERS=4 # number of expert layers
export N_FILTERS=32 # number of hidden filters (aka channel size)
export N_EXPERTS=3 # number of experts
export BATCH_SIZE=2 # other configs assume default is 2 so lets make it explicit
export IN_STEPS=1 # timesteps stacked as the model input
export OUT_STEPS=9 # other configs assume default is 9 output steps (chunk length 10)
export USE_NORMALIZED_MOE=1 # importance normalization
