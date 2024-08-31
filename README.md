# MOR_MoE
MOR-NeuralOP Mixture of Experts

**First use mamba/conda to import the env file: uqops+proxy.yaml**

To run, you need either use:
* The exercises.ipynb notebook, exercise 3 (to play with it)
* Or channel.py (for big jobs)
* Or job.slurm (for bigger kahuna jobs)

Important Files:
* JHTDB_sim_op.py: Integration of PDE solver & POU_net + dataset code
* POU_net.py: The current POU_net code & FieldGatingNet
* MOR_Operator.py: MOR_layer + MOR_Operator modules
* lightning_utils.py: various utilities
* channel.py: the entrypoint
