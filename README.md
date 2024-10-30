# MOR_MoE
MOR-NeuralOP Mixture of Experts

**First use mamba/conda to import the env file: uqops+proxy.yaml**

To run, you need either use:
* The exercises.ipynb notebook, exercise 3 (to play with it, e.g. debug new features)
* Or channel.py (for big jobs, e.g. on CEE)
* **Or job.slurm** (for bigger kahuna jobs)

Model Training Notes:
* job.slurm also contains many comments to explain the configuration settings
* *You must* set **MAX_EPOCHS** to accurately reflect the estimated number of epochs that will elapse for a particular job.
  * If you use too few, the job will waste possible training time. If you set too many the OneCycle schedule will be poorly calibrated!
  * You can use the recorded reference points in the comments to estimate the epochs for a given job.
* After training you need to navigate to your chosen model in lightning logs and consolidate the checkpoint
  * e.g. cd ~/MOR_MoE/lightning_logs/YOUR_MODEL_NAME/12345/checkpoints epoch=405-step=14902.ckpt



Important Files:
* JHTDB_sim_op.py: Integration of PDE solver & POU_net + dataset code
* POU_net.py: The current POU_net code & FieldGatingNet (also includes PPOU_net, which is to say VI version)
* MOR_Operator.py: MOR_layer + MOR_Operator modules
* model_agnostic_BNN.py: code for 
* channel.py: the entrypoint


* lightning_utils.py: various utilities (e.g. code for a CNN network)
