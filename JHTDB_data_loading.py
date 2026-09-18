from __future__ import annotations
import os
import torch
import h5py
import numpy as np
from glob import glob
import pytorch_lightning as L
from copy import copy
from typing import Any
_get_strided_length = lambda length, stride: (length-1)//stride+1 # -1 for last TS, +1 for first TS

class DatasetMetaData:
    def __init__(self, Lx: float, Ly: float, Lz: float, nu: float, dt: float, n_steps_per_flow_through_native: int):
        vars(self).update(locals()); del self.self # save configuration args settings

    def __getattr__(self, name: str) -> Any:
        try: super().__getattr__(name)
        except AttributeError:
            if 'nx' in vars(self): raise # already adapted to stride
            else: raise AttributeError(f'{name} is not a valid attribute of {type(self).__name__} (you may need to call self.adapt_to_stride(dataset) to finish construction).')

    def adapt_to_stride(self, dataset: JHTDB_Channel|IUFNO_Channel) -> DatasetMetaData:
        ''' Adds fields: time_stride, n_steps_per_flow_through, field_size, nx, ny, nz extracted from the dataset object.'''
        new = copy(self)
        new.time_stride = dataset.time_stride
        new.dt *= dataset.time_stride
        new.n_steps_per_flow_through = _get_strided_length(self.n_steps_per_flow_through_native, dataset.time_stride)
        IC_0, _ = dataset[0]
        new.field_size = tuple(IC_0.shape[1:])
        new.nx, new.ny, new.nz = new.field_size
        return new

    def with_eval_time_stride(self, eval_time_stride: int) -> DatasetMetaData:
        assert eval_time_stride % self.time_stride == 0, f'{eval_time_stride=} must be a multiple of {self.time_stride=}'
        extra_time_stride = eval_time_stride // self.time_stride
        new = copy(self)
        new.dt *= extra_time_stride
        new.n_steps_per_flow_through = _get_strided_length(self.n_steps_per_flow_through, extra_time_stride)
        new.time_stride = eval_time_stride
        if eval_time_stride > max(1, round(0.2 * self.n_steps_per_flow_through_native)):
            import warnings
            warnings.warn(f'{eval_time_stride=} cannot resolve 0.2 FTT ({0.2 * self.n_steps_per_flow_through_native} unstrided steps)')
        return new

    def __repr__(self):
        items = [f'{k}={v}' for k, v in sorted(vars(self).items()) if not k.startswith('_')]
        return f'{type(self).__name__}({", ".join(items)})'

    @property
    def simulator_kwd_args(self): # while its true this couples with the _Sim constructor, it's still a useful meta-data operation & simpler legacy option
        ''' Physics kwargs for _Sim. Requires adapt_to_stride (nx, ny, nz). Solver flags stay with the caller.
            construction example: `_Sim(**DatasetMetaData.simulator_kwd_args, **other_simulator_flags)`'''
        return dict(nx=self.nx, ny=self.ny, nz=self.nz, Lx=self.Lx, Ly=self.Ly, Lz=self.Lz, dt=self.dt, nu=self.nu)

# Verified to work: 8/23/24
class JHTDB_Channel(torch.utils.data.Dataset):
    '''
    Dataset for the JHTDB autoregressive problem... It is not possible to make
    this predict everything at once because that would make the dataset size=1.
    '''
    _default_meta_data = DatasetMetaData(Lx=8*np.pi, Ly=2.0, Lz=3*np.pi, nu=5e-5, dt=0.0065, n_steps_per_flow_through_native=4000)

    def __init__(self, path:str, time_chunking=5, stride:int|list|tuple=1, time_stride:int=1):
        self.path=path
        self.time_chunking=time_chunking
        self._time_stride=time_stride # ready-only
        assert type(time_stride) is int
        if type(stride) in [int,float]: stride=[stride]*3
        else: assert len(stride)==3 # we will not pool time because it breaks PDE timestep & stability and pytorch cannot do it easily
        scale_factor = tuple(1/np.asarray(stride).astype(float))
        self._pool = lambda x: torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='area')
        # comparable to torch.nn.AvgPool3d(stride) but supports fractional stride

        self._split_start_proportion = 0
        self._split_end_proportion = 1.0 # exclusive 1.0=length of dataset

        self.dataset_meta_data = self._default_meta_data.adapt_to_stride(self)

    @property
    def time_stride(self): return self._time_stride

    def split(self, proportion: float):
        assert 0 <= proportion <= 1, 'proportion must be between 0 and 1'
        from copy import copy
        base_start = self._split_start_proportion
        base_end = self._split_end_proportion
        split_point = base_start + proportion * (base_end - base_start)

        start_dataset = copy(self)
        start_dataset._split_start_proportion = base_start
        start_dataset._split_end_proportion = split_point

        end_dataset = copy(self)
        end_dataset._split_start_proportion = split_point
        end_dataset._split_end_proportion = base_end

        return start_dataset, end_dataset

    @property
    def _split_file_index_range(self) -> tuple[int, int]:
        num_files = len(glob(f'{self.path}/*.h5'))
        start_index = int(num_files * self._split_start_proportion)
        end_index = int(num_files * self._split_end_proportion)
        return start_index, end_index

    def __len__(self):
        # NOTE: for overlapping chunks, you'd use num_files - ((self.time_chunking - 1) * self.time_stride)
        # GOTCHA: it might be better to apply the data augmentation randomly rather than changing the length (for VI consistency);
        # that being said, it wouldn't really work with dataset preloading, and I'm not sure how it would interact with existing offseting.
        start_index, end_index = self._split_file_index_range
        num_files = end_index - start_index
        base_blocks = num_files // (self.time_chunking * self.time_stride)  # full blocks only
        return base_blocks * self.time_stride  # one sample per offset per block

    # Time stride verified to work: 11/18/25
    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f'Index {index} is out of range for dataset of length {len(self)}')

        files = []
        velocity_fields = []
        start_index, end_index = self._split_file_index_range
        offset = index % self.time_stride
        index = index // self.time_stride
        # NOTE: for overlapping chunks, you'd use range(index, index+self.time_chunking*self.time_stride, self.time_stride)
        for i in range(index*self.time_chunking*self.time_stride, (index+1)*self.time_chunking*self.time_stride, self.time_stride):
            i+=1 + offset + start_index # 1-based indexing + offset to utilize all data with time stride + start index to skip split files
            assert i <= end_index, f'File index: {i} is above maximum {end_index}'
            try: files.append(h5py.File(f'{self.path}/channel_t={i}.h5', 'r')) # keep open for stacking
            except OSError as e:
                if 'unable to open' in str(e).lower():
                    raise OSError(f'Unable to open file: "{self.path}/channel_t={i}.h5"')
                else: raise
            velocity_fields.append(files[-1][f'Velocity_{i:04}']) # :04 zero pads to 4 digits
        velocity_fields = torch.as_tensor(np.stack(velocity_fields).T) # reverse dimensions order [T,Z,Y,X,C] --> [C,X,Y,Z,T]
        velocity_fields = self._pool(velocity_fields.moveaxis(-1,0)).moveaxis(0,-1) # time dimension is (temporarily) treated as batch dimension
        velocity_fields = velocity_fields.float() # make sure to use single precision! (after pooling) because double is too expensive!!

        # IC_0.shape=[C,X,Y,Z] e.g. torch.Size([3, 103, 26, 77])
        # Sol_0.shape=[C,X,Y,Z,T] e.g. torch.Size([3, 103, 26, 77, 9])
        return velocity_fields[...,0], velocity_fields[...,1:] # X=IC, Y=sol

## vor=vorticity
## Dwyer: 21x400x32x33x16x4 (groups, time, x, y, z, channels)
## channel flow variables are: u,v,w and p.
## -------------------------------------------------
#vor_data = np.load('./IUFNO-CHL/data_chl_re180/data_mave.npy')
#vor_data = vor_data[...,0:3]
#vor_data = vor_data[0:20,...]
##-------------------------------------------------
# NOTE: not possible to directly reuse JHTDB_Channel because group boundaries are discontinuous
class IUFNO_Channel(JHTDB_Channel):
    '''IUFNO channel flow. Returns (IC, future frames) like JHTDB_Channel.'''

    @property
    def _default_meta_data(self) -> DatasetMetaData:
        if not ('re180' in self.path or 're590' in self.path): raise ValueError(f'{self.path=} is not a registered IUFNO dataset')
        nu = 1/4200 if 're180' in self.path else 1/16800 # apparently this and resolution are the differences
        return DatasetMetaData(Lx=4*np.pi, Ly=2.0, Lz=4*np.pi/3, nu=nu, dt=1.0, n_steps_per_flow_through_native=19)
        # NOTE: Technically 400/21.25=18.82352941 steps per flow through, and 21.25 flow throughs per (400 step) group (we rounded)

    def __init__(self, path:str, time_chunking=5, stride:int|list|tuple=1, time_stride:int=1):
        self._data = np.load(path, mmap_mode='r')[..., :3]  # [G,T,X,Y,Z,C=uvw]

        try: # Y-profile mean for adding back to fluctuations
            ave = np.load(os.path.join(os.path.dirname(path), 'data_ave.npy'))[0, ..., :3] # [T=1,X=1,Y,Z=1,C=uvw]: drop group
            self.mean_field = torch.as_tensor(ave).permute(4, 1, 2, 3, 0).float()  # [C,X=1,Y,Z=1,T=1]
        except FileNotFoundError:
            self.mean_field = 0

        super().__init__(path, time_chunking, stride, time_stride)

    @property
    def n_groups(self) -> int:
        return self._data.shape[0]

    @property
    def _split_group_index_range(self) -> tuple[int, int]:
        return int(self.n_groups * self._split_start_proportion + 0.5), \
            int(self.n_groups * self._split_end_proportion + 0.5)

    @property
    def _samples_per_group(self) -> int:
        n_times = self._data.shape[1] # 400
        base_blocks, remainder = divmod(n_times, self.time_chunking * self.time_stride)
        extra = max(0, remainder - (self.time_chunking - 1) * self.time_stride)
        return base_blocks * self.time_stride + extra

    def __len__(self):
        start, end = self._split_group_index_range
        return (end - start) * self._samples_per_group

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f'Index {index} is out of range for dataset of length {len(self)}')

        group_index, local_index = divmod(index, self._samples_per_group)
        group_index += self._split_group_index_range[0] # start
        block, offset = divmod(local_index, self.time_stride) # time stride multiplies the number of chunks per block
        t0 = block * self.time_chunking * self.time_stride + offset
        t1 = t0 + self.time_chunking * self.time_stride # consistent with JHTDB_Channel: `(index+1)*self.time_chunking*self.time_stride` ends at next block boundary
        chunk = np.array(self._data[group_index, t0:t1:self.time_stride])  # [T,X,Y,Z,C], copy off mmap
        fields = torch.as_tensor(chunk).permute(4, 1, 2, 3, 0)  # [C,X,Y,Z,T]
        fields = fields + self.mean_field # add y-profile mean back (broadcast over X,Z,T)
        fields = self._pool(fields.moveaxis(-1, 0)).moveaxis(0, -1).float()
        return fields[..., 0], fields[..., 1:]

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
                 stride: int|list|tuple=1, long_horizon: int=400, train_proportion: float=0.8,
                 dataset_type: type[JHTDB_Channel|IUFNO_Channel] = JHTDB_Channel,
                 preload_datasets: bool=True, fast_dataloaders: bool=False):
        assert 0 < train_proportion < 1, 'train_proportion must be between 0 and 1'
        super().__init__()
        self.save_hyperparameters()

        # fence post counting: -1 for the last time step, +1 for the first time step
        long_horizon = max(2, (long_horizon-1) // time_stride + 1) # must be at least 2 to have input and output
        vars(self).update(locals()); del self.self # save configuration args settings
        self.setup('peek') # trivial setup to expose basic dataset info

    @property
    def _fast_dataloader_kwd_args(self): # Optional faster dataloaders (uses more memory)
        return {'num_workers': 1 if self.preload_datasets else 8, 'persistent_workers': self.preload_datasets} if self.fast_dataloaders else {}

    def setup(self, stage: str='fit'):
        ''' if stage=='peek': do not preload the dataset,
        also setup('peek') is automatically called in the constructor
        since it doesn't cost anything '''

        # Build datasets mirroring the main() logic
        self.dataset = self.dataset_type(self.dataset_path, time_chunking=self.time_chunking, stride=self.stride, time_stride=self.time_stride)
        dataset_long_horizon = self.dataset_type(self.dataset_path, time_chunking=self.long_horizon, stride=self.stride, time_stride=self.time_stride)
        if len(self.dataset)<self.batch_size: raise ValueError(f'Dataset files missing! {self.dataset_path=}')

        # this splitting is necessary because we need to split on the file level, not the coarse time chunk level
        self.val_long_horizon_dataset = dataset_long_horizon.split(self.train_proportion)[1]

        # this kind of splitting is better for timeseries so that we can measure true extrapolation performance
        self.train_dataset, self.val_dataset = self.dataset.split(self.train_proportion)

        if stage=='peek': # sanity checks
            print(f'{len(self.dataset)=}\n{len(self.train_dataset)=}\n{len(self.val_dataset)=}')
            print(f'{len(self.val_long_horizon_dataset)=}')
        assert min(len(self.dataset), len(self.train_dataset), len(self.val_dataset), len(self.val_long_horizon_dataset))>0, f'Empty datasets! {self.dataset_path=}'

        if stage!='peek' and self.preload_datasets: # preload the datasets
            self.val_long_horizon_dataset = preload_dataset(self.val_long_horizon_dataset)
            self.train_dataset = preload_dataset(self.train_dataset)
            self.val_dataset = preload_dataset(self.val_dataset)

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

    def meta_data(self, eval_time_stride: int | None = None) -> DatasetMetaData:
        md = self.dataset.dataset_meta_data
        if eval_time_stride is None: return md
        return md.with_eval_time_stride(eval_time_stride)

    @property
    def u_b(self):
        # u_b = real_channel_flow[0].mean((0,2,3)) (for manual advection term)
        full_field = self.load_full_dataset_field()
        return full_field[0].mean((0,2,3))

    def load_full_dataset_field(self, time_stride:int=None):
        ''' Used for comparing full learned simulations to DNS '''
        import os
        if time_stride is None: time_stride = self.time_stride
        cache_path = f'{self.dataset_path}/full_field_cache_{self.stride=}_{time_stride=}.pt'
        if os.path.exists(cache_path): # almost x3 faster!
            print('loading from cache!')
            return torch.load(cache_path)

        dataset = self.dataset_type(self.dataset_path, time_chunking=1, stride=self.stride)
        if isinstance(dataset, IUFNO_Channel):
            dataset = dataset.split((dataset.n_groups-1)/dataset.n_groups)[1] # keep only the last group
        dataset = torch.utils.data.Subset(dataset, range(0,len(dataset),time_stride))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8)

        full_field = torch.stack([x.squeeze() for x, _ in data_loader], axis=-1)
        print(f'{full_field.shape=}, {full_field.device=}')
        if self.preload_datasets and os.path.isdir(os.path.dirname(cache_path)):
            torch.save(full_field, cache_path)
        return full_field
