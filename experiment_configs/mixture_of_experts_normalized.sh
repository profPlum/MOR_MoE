# these are already the defaults but written explicitly
export N_LAYERS=4 # number of expert layers
export N_FILTERS=32 # number of hidden filters (aka channel size)
export N_EXPERTS=3 # number of experts

# MoE specific
export TIME_CHUNKING=2 # disable recursive steps
export BATCH_SIZE=18 # =2*(10-1)/(2-1) bigger b/c we removed the recursive steps
export USE_NORMALIZED_MOE=1 # use gating weight importance regularization
