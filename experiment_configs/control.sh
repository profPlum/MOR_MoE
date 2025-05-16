# These settings disable everything i.e. auto regressive timestepping & mixture of experts (which is to say it replaces mixture with one big expert)
export TIME_CHUNKING=2 # 9 is default (& the max for 10x A100 GPUs), 2 is the minimum (i.e. it disables auto-regressive timestepping feature)
export N_LAYERS=6 # number of expert layers, default_params = 2*proj_layer + 2*full_layer, full_layer >> proj_layer so we double that = 2*proj_layer + 4*full_layer = 6 layers
export N_FILTERS=33 # (default=32) + 1 to correct for not doubling projection layers (works almost perfectly!)
export N_EXPERTS=1 # number of experts specifying one implies not even the zero expert is used
export BATCH_SIZE=13 # really should be 13.5 but whatever bigger b/c we removed the recursive steps
