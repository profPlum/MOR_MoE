# MOR_MoE
MOR-NeuralOP Mixture of Experts

Setup:
* (on Kahuna) Set Cuda Override `export CONDA_OVERRIDE_CUDA="11.8"`
* **use mamba/conda to import the env file: uqops.yaml**
* use the master branch

To run, you need either use:
* **(primarily) job.slurm** (for bigger kahuna jobs)
* Or train.py (for big jobs, e.g. on CEE)
* Or The `exercises.ipynb` notebook, exercise 3 (to play with it, e.g. debug new features)

Model Training Notes:
* GOTCHA: It is best to use **exactly** the maximum number of modes for training because there is a huge memory/time optimization that works only when this is true.
* *You must* set **MAX_EPOCHS** to accurately reflect the estimated number of epochs that will elapse for a particular job!
  * If you use too few, the job will waste possible training time. If you set too many the OneCycle schedule will be poorly calibrated!
  * You can use the recorded reference points in the comments to estimate the epochs for a given job.
* After training you need to navigate to your chosen model in lightning logs and consolidate the checkpoint (otherwise you can't use it)
  * e.g. `cd ~/MOR_MoE/lightning_logs/YOUR_MODEL_NAME/123456/checkpoints; python -m pytorch_lightning.utilities.consolidate_checkpoint epoch=405-step=14902.ckpt`
* Resuming training on a model can be a useful way to turn an ok model into a better model, these are the guidelines & instructions for doing so:
  * consolidate the checkpoint of the model you want to resume training for (see previous bullet)
  * set `CKPT_PATH`, e.g.: `export CKPT_PATH='/home/dsdeigh/MOR_MoE/lightning_logs/PARTIAL_VI_RLoP_Extension/477516/checkpoints/epoch=206-step=7452.ckpt.consolidated'`
  * Consider using `export OPTIM='RAdam'` and `export RLoP=1` this will shorten the long warmup phase from OneCycle and quickly adjust learning rate to something appropriate for late-stage training
* Adjust `TIME_CHUNKING`, e.g. `export TIME_CHUNKING=9` for 10 nodes, or `export TIME_CHUNKING=10` for 20 nodes.
* Finally after you've handled these things do `sbatch job.slurm`

Model Evaluation Notes:
* *Weights and Biases* is important for evaluation, but these are the metrics which are most important:
  * All val metrics with data_loader_idx=1 postfix (the long-horizon dataset) are more important than data_loader_idx=0 (the short horizon dataset)
  * val_loss is very important too because it incorporates UQ and prediction quality (again data_loader_idx=1 is more important)
  * grad_inf_norm can be important to assess the stability of the model
  * the pytorch profiler tab can be useful for checking training performance
  * val_R^2 especially for the data_loader_idx=1 is very important (arguably the most)
* Aside from Weights and Biases, most important model evaluation code is inside **JHTDB_operator.ipynb**, this notebook can:
  * perform learned simulations & comparison with DNS baseline
  * create 3d expert partition figures
  * create energy spectrum figures for comparing learned simulation to DNS

*Important* Files:
* JHTDB_sim_op.py: Integration of PDE solver & POU_net + dataset code (also includes some VI code)
* POU_net.py: The current POU_net code & FieldGatingNet (also includes PPOU_net, which is to say VI version)
* MOR_Operator.py: MOR_layer + MOR_Operator modules
* model_agnostic_BNN.py: code for model agnostic VI
* grid_figures.py: code for the pretty 4d grid figures I make (e.g. the simulation)
* train.py: the (python) entrypoint (uses env variables for CLI)
* job.slurm: the (slurm) entrypoint
* JHTDB_operator.ipynb: the primary model evaluation notebook
