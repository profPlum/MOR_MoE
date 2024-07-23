import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as L
from lightning_utils import BasicLightningRegressor
import MOR_Operator

# Verified to work: 7/18/24
# Double verified to work (and reproduce specific partition)
# Triple verified to work (with higher dimensionalities/n_inputs)
class FieldGatingNet(BasicLightningRegressor):
    """
    Essentially a Conv MLP that outputs class probabilities across the field.
    Currently it is not actually a function of the field, just it's size (& the corresponding positions).
    """
    def __init__(self, n_inputs, n_experts, n_filters=20):
        super().__init__()
        assert n_inputs in [1,2,3]
        self.n_inputs=n_inputs
        ConvLayer = [nn.Conv1d, nn.Conv2d, nn.Conv3d][n_inputs-1]
        self._gating_net = nn.Sequential(
          ConvLayer(n_inputs,n_filters,1),
          nn.ReLU(),
          ConvLayer(n_filters,n_filters,1),
          nn.ReLU(),
          ConvLayer(n_filters,n_filters,1),
          nn.ReLU(),
          ConvLayer(n_filters,n_experts,1),
          nn.Softmax(dim=1)
        )
    def _make_positional_encodings(self, shape):
        # Create coordinate grids using torch.meshgrid
        assert len(shape)==self.n_inputs
        coords = [torch.linspace(0,1,steps=dim) for dim in shape]
        mesh = torch.meshgrid(*coords)
        pos_encodings = torch.stack(mesh)[None] # [None] adds batch dim
        return pos_encodings.to(self.device)
    def forward(self, X):
        pos_encodings = self._make_positional_encodings(X.shape[-self.n_inputs:])
        return self._gating_net(pos_encodings)

# NOTE: The new POU_net should actually generalize the original!
# We only exported the gating net & the miracle is that the forward method still works!!
class POU_net(BasicLightningRegressor):
    def __init__(self, make_expert=MOR_Operator.MOR2dOperator, n_experts=5, n_inputs=2, make_gating_net: type=FieldGatingNet,
                 lr=0.001, L2_weight=1.0, L2_patience=10, L2_decay=0.9, T_max=25):
        super().__init__()
        # NOTE: setting n_experts=n_experts+1 inside the gating_net implicitly adds a "ZeroExpert"
        self.gating_net=make_gating_net(n_inputs=n_inputs, n_experts=n_experts+1) # supports n_inputs!=2
        self.experts=nn.ModuleList([make_expert() for i in range(n_experts)])
        vars(self).update(locals()); del self.self
    def configure_optimizers(self):
        param_groups = [{'params': self.experts.parameters(), 'weight_decay': self.L2_weight},
                        {'params': self.gating_net.parameters()}] #L2 for experts only
        self.experts_optim = torch.optim.Adam(param_groups, lr=self.lr)
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.experts_optim, T_0=self.T_max, T_mult=2)
        #lr_schedule=torch.optim.lr_scheduler.CosineAnnealingLR(self.experts_optim, T_max=self.T_max)
        return [self.experts_optim], [lr_schedule]
    # Verified to work 7/19/24
    def forward(self, X):
        X = torch.as_tensor(X).to(self.device)
        gating_weights = self.gating_net(X)
        prediction = 0
        for i, expert in enumerate(self.experts):
            prediction = prediction + gating_weights[:,i]*expert(X)
        return prediction
    def on_train_start(self):
        self._lowest_loss = float('inf') # for L2 patience
        self._n_stag_steps = 0
    # Verified to work 7/19/24
    def _decay_L2(self, loss):
        """ Decay L2 like original POU_net, but maybe not necessary for Operator MoE? """
        if self._lowest_loss>loss.item():
            self._lowest_loss=loss.item()
            self._n_stag_steps=0
        else:
            self._n_stag_steps += 1
            if self._n_stag_steps%self.L2_patience==0:
                self.experts_optim.param_groups[0]['weight_decay'] *= self.L2_decay
        self.log('L2_coef', float(self.experts_optim.param_groups[0]['weight_decay']), prog_bar=True)
    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch)
        self._decay_L2(loss)
        return loss