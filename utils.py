''' Debugging utilities '''

import torch
import numpy as np

fields = lambda x: [name for name in vars(x) if not name.startswith('_')] # utility to see public fields

def tshow(x): # utility to "show" tensors without overwhelming people
    try: x = x.cpu().detach()
    except: pass
    x = np.asarray(x)
    print('='*30)
    print(f'{x.shape=}\n{x.dtype=}\n{x.min()=:.2e}\n{x.max()=:.2e}\n{x.mean()=:.2e}\n{x.std()=:.2e}')
    print('='*30)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def LINE(up=0):
    import inspect, os
    frame = inspect.currentframe().f_back
    for _ in range(up): frame = frame.f_back
    fn = os.path.basename(frame.f_code.co_filename)
    return f'pid: {os.getpid()}, filename: {fn}, function: {frame.f_code.co_name}(), line: {frame.f_lineno}'

import time
def check_point(msg:str):
    check_point.num+=1
    new_time = time.time()
    duration = new_time-check_point.last_time
    msg = f'Checkpoint #{check_point.num} ({LINE(1)}): {msg}\nSeconds since last checkpoint: {duration}'
    print(msg, flush=True)
    check_point.last_time=new_time
check_point.num=0
check_point.last_time=time.time()

from contextlib import contextmanager

# This is a different flavor of check_point
@contextmanager
def profile_it(msg: str):
    try:
        profile_it.num+=1
        start_time = time.time()
        yield
    finally:
        duration = time.time()-start_time
        msg = f'Profile #{profile_it.num} ({LINE(1)}): {msg}\nDuration in Seconds: {duration}'
        print(msg, flush=True)
profile_it.num=0

# the simpler ways... now deprecated for consistent/recorded scaling
np_normalize = lambda array: (array-array.mean())/array.std()

class StopExecution(Exception):
    def _render_traceback_(self):
        return []

def clear_cache():
    import torch, gc
    while gc.collect(): pass
    torch.cuda.empty_cache()

def nvidia_smi(message='', clear_mem=False, verbose=False):
    import sys
    if message: message+='\n'
    if clear_mem:
        message+='(clearing cache!)\n'
        clear_cache()
    line = LINE(1) # pid, fn, func, lineno
    full_msg = '\n'+'-'*50+ f'\n{message}from device: {torch.cuda.current_device()}, {line}\n'+'-'*50
    try: full_msg += f"\nGPU Utilization: {torch.cuda.utilization()}%"
    except: print('Warning: unable to display GPU utilization try installing "pynvml" via pip', file=sys.stderr)
    full_msg += "\ntorch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated()/1024/1024/1024)
    full_msg += "\ntorch.cuda.max_memory_allocated: %fGB"%(torch.cuda.max_memory_allocated()/1024/1024/1024)
    if verbose:
        full_msg+="\ntorch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved()/1024/1024/1024)
        full_msg+="\ntorch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved()/1024/1024/1024)
    print(full_msg, flush=True)
    torch.cuda.reset_peak_memory_stats() # reset since we reported it
    #if verbose: print(torch.cuda.memory_summary()) # way too much info
