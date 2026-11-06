# these are already the defaults but written explicitly
export N_LAYERS=4 # number of expert layers
export N_FILTERS=32 # number of hidden filters (aka channel size)
export N_EXPERTS=3 # number of experts

# MoE specific
export IN_STEPS=1 OUT_STEPS=1 # disable recursive steps (chunk length was 2)
export BATCH_SIZE=18 # =2*(10-1)/(2-1) bigger b/c we removed the recursive steps
export USE_NORMALIZED_MOE=0 # must be explicitly false for this comparison
