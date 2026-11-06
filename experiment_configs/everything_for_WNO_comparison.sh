# these are already the defaults but written explicitly
export N_LAYERS=4 # number of expert layers
export N_EXPERTS=3 # number of experts
export BATCH_SIZE=2 # other configs assume default is 2 so lets make it explicit
export IN_STEPS=1 # timesteps stacked as the model input
export OUT_STEPS=9 # other configs assume default is 9 output steps (chunk length 10)
export USE_NORMALIZED_MOE=1 # importance normalization

export USE_WNO3D_EXPERTS=0 # To use WNO3d experts: set USE_WNO3D_EXPERTS=1 and USE_CNN_EXPERTS=0
export STRIDE='[1.609,1.625,1.203]' FAST_DATALOADERS=1 # 64x16x64
export MAX_EPOCHS=800
export N_FILTERS=36

