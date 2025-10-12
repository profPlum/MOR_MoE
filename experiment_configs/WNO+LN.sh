# WNO3d experts?
export USE_WNO3D_EXPERTS=1 # To use WNO3d experts: set USE_WNO3D_EXPERTS=1 and USE_CNN_EXPERTS=0
export WNO3D_LEVEL=1 # wavelet decomposition level (default 2, higher = more detailed wavelet analysis)
export N_EXPERTS=1 N_LAYERS=4
export STRIDE='[1.609,1.625,1.203]' FAST_DATALOADERS=1
export MAX_EPOCHS=800
export BATCH_SIZE=2
export OUT_NORM_GROUPS=1 HIDDEN_NORM_GROUPS=1
