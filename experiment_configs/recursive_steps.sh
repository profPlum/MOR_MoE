# these settings relpace mixture of experts with one big expert
export N_LAYERS=6 # number of expert layers, default_params = 1*proj_layer + 3*full_layer, full_layer >> proj_layer so we double that = 1*proj_layer + 6*full_layer ~= 7 layers
export N_FILTERS=36 # (default=32) + 1 to correct for not doubling projection layers (works almost perfectly!)
export N_EXPERTS=1 # number of experts specifying one implies not even the zero expert (or gating network) is used (i.e. disable mixture of experts
export BATCH_SIZE=2 # other configs assume default is 3 so lets make it explicit
export TIME_CHUNKING=10 # other configs assume default is 10 so lets make it explicit
