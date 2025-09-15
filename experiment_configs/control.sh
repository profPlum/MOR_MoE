# These settings disable everything i.e. auto regressive timestepping & mixture of experts (which is to say it replaces mixture with one big expert)
export TIME_CHUNKING=2 # 10 is default, 2 is the minimum (i.e. it disables auto-regressive timestepping feature)
export N_LAYERS=6 # number of expert layers, default_params = 1*proj_layer + 3*full_layer, full_layer >> proj_layer so we double that = 1*proj_layer + 6*full_layer = 7 layers
export N_FILTERS=36 # (default=32) + 1 to correct for not doubling projection layers (works almost perfectly!)
export N_EXPERTS=1 # number of experts specifying one implies not even the zero expert is used
export BATCH_SIZE=18 # =2*(10-1)/(2-1) bigger b/c we removed the recursive steps
