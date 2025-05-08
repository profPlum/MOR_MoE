# these are already the defaults but written explicitly
export TIME_CHUNKING=9 # 9 is default (b/c it is max for x10 A100s)
export N_LAYERS=4 # number of expert layers
export N_FILTERS=32 # number of hidden filters (aka channel size)
export N_EXPERTS=3 # number of experts
