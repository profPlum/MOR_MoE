import torch
import h5py
import numpy as np
from glob import glob
import pytorch_lightning as L
import math

# Verified to work: 8/23/24
class JHTDB_Channel(torch.utils.data.Dataset):
    '''
    Dataset for the JHTDB autoregressive problem... It is not possible to make
    this predict everything at once because that would make the dataset size=1.
    '''
    def __init__(self, path:str, time_chunking=5, stride:int|list|tuple=1, time_stride:int=1):
        self.path=path
        self.time_chunking=time_chunking
        self.time_stride=time_stride
        assert type(time_stride) is int
        if type(stride) in [int,float]: stride=[stride]*3
        else: assert len(stride)==3 # we will not pool time because it breaks PDE timestep & stability and pytorch cannot do it easily
        scale_factor = tuple(1/np.asarray(stride).astype(float))
        self.pool = lambda x: torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='area')
        # comparable to torch.nn.AvgPool3d(stride) but supports fractional stride

    def __len__(self):
        num_files = len(glob(f'{self.path}/*.h5'))
        base_blocks = num_files // (self.time_chunking * self.time_stride)  # full blocks only
        return base_blocks * self.time_stride  # one sample per offset per block

    # Time stride verified to work: 11/18/25
    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f'Index {index} is out of range for dataset of length {len(self)}')

        files = []
        velocity_fields = []
        offset = index % self.time_stride
        index = index // self.time_stride
        for i in range(index*self.time_chunking*self.time_stride, (index+1)*self.time_chunking*self.time_stride, self.time_stride):
            i+=1 + offset # 1-based indexing + offset to utilize all data with time stride
            try: files.append(h5py.File(f'{self.path}/channel_t={i}.h5', 'r')) # keep open for stacking
            except OSError as e:
                if 'unable to open' in str(e).lower():
                    raise OSError(f'Unable to open file: "{self.path}/channel_t={i}.h5"')
                else: raise
            velocity_fields.append(files[-1][f'Velocity_{i:04}']) # :04 zero pads to 4 digits
        velocity_fields = torch.as_tensor(np.stack(velocity_fields).T) # reverse dimensions order [T,Z,Y,X,C] --> [C,X,Y,Z,T]
        velocity_fields = self.pool(velocity_fields.moveaxis(-1,0)).moveaxis(0,-1) # time dimension is (temporarily) treated as batch dimension
        velocity_fields = velocity_fields.float() # make sure to use single precision! (after pooling) because double is too expensive!!

        # IC_0.shape=[C,X,Y,Z] e.g. torch.Size([3, 103, 26, 77])
        # Sol_0.shape=[C,X,Y,Z,T] e.g. torch.Size([3, 103, 26, 77, 9])
        return velocity_fields[...,0], velocity_fields[...,1:] # X=IC, Y=sol

# preserves random state (verified to work: 9/24/25)
def preload_dataset(dataset):
    if isinstance(dataset, torch.utils.data.TensorDataset):
        return dataset

    Xs, ys = [], []
    for i in range(len(dataset)):
        X, y = dataset[i]
        Xs.append(X)
        ys.append(y)
    Xs, ys = torch.stack(Xs), torch.stack(ys)
    print(f'dataset min={min(Xs.min(),ys.min())}, max={max(Xs.max(),ys.max())}')
    return torch.utils.data.TensorDataset(Xs, ys)

class JHTDBDataModule(L.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size: int, time_chunking: int, time_stride: int=1,
                 stride: int|list|tuple=1, long_horizon: int=100, train_proportion: float=0.8, fast_dataloaders: bool=False):
        assert 0 < train_proportion < 1, 'train_proportion must be between 0 and 1'
        super().__init__()
        self.save_hyperparameters()
        vars(self).update(locals()); del self.self # save configuration args settings
        self.setup('peek') # trivial setup to expose basic dataset info

    @property
    def _fast_dataloader_kwd_args(self): # Optional faster dataloaders (uses more memory)
        return {'num_workers': 1, 'persistent_workers': True} if self.fast_dataloaders else {}

    def setup(self, stage: str='fit'):
        ''' if stage=='peek': do not preload the dataset,
        also setup('peek') is automatically called in the constructor
        since it doesn't cost anything '''

        # Build datasets mirroring the main() logic
        self.dataset = JHTDB_Channel(self.dataset_path, time_chunking=self.time_chunking, stride=self.stride, time_stride=self.time_stride)
        if stage!='peek': self.dataset = preload_dataset(self.dataset)
        dataset_long_horizon = JHTDB_Channel(self.dataset_path, time_chunking=self.long_horizon, stride=self.stride, time_stride=self.time_stride)

        self.val_long_horizon_dataset = torch.utils.data.Subset(dataset_long_horizon, torch.arange(math.ceil(len(dataset_long_horizon)*self.train_proportion), len(dataset_long_horizon)))
        if stage!='peek': self.val_long_horizon_dataset = preload_dataset(self.val_long_horizon_dataset)

        # this kind of splitting is better for timeseries so that we can measure true extrapolation performance
        self.train_dataset = torch.utils.data.Subset(self.dataset, torch.arange(int(len(self.dataset)*self.train_proportion)))
        self.val_dataset = torch.utils.data.Subset(self.dataset, torch.arange(int(len(self.dataset)*self.train_proportion), len(self.dataset)))

        if stage=='peek':
            print(f'{len(self.dataset)=}\n{len(self.train_dataset)=}\n{len(self.val_dataset)=}')
            print(f'{len(self.val_long_horizon_dataset)=}')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True, drop_last=True, **self._fast_dataloader_kwd_args)

    def val_dataloader(self, batch_size=None):
        # Derived quantities for long-horizon validation
        long_horizon_multiplier = self.long_horizon / self.time_chunking
        long_horizon_batch_size = max(1, int(self.batch_size / long_horizon_multiplier))

        if batch_size is None: batch_size=int(long_horizon_batch_size*long_horizon_multiplier)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, pin_memory=True, **self._fast_dataloader_kwd_args)
        val_long_loader = torch.utils.data.DataLoader(self.val_long_horizon_dataset, batch_size=long_horizon_batch_size, pin_memory=True, **self._fast_dataloader_kwd_args)
        return {'val': val_loader, 'long_horizon': val_long_loader}

    @property
    def field_size(self):
        IC_0, Sol_0 = self.train_dataset[0]
        print(f'{IC_0.shape=}\n{Sol_0.shape=}')

        field_size = list(IC_0.shape[1:])
        print(f'{field_size=}')
        return field_size