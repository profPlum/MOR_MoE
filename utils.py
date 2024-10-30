import torch
from torch import nn

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

def report_cuda_memory_usage(message='', clear=True, verbose=False):
    if message: message+='\n'
    if clear:
        message+='(clearing cache!)\n'
        clear_cache()
    line = LINE(1) # fn, func, lineno
    full_msg = '\n'+'-'*50+'\n' + f'{message}from {line}'+'\n'+'-'*50
    full_msg += "\ntorch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024)
    full_msg += "\ntorch.cuda.max_memory_allocated: %fGB"%(torch.cuda.max_memory_allocated(0)/1024/1024/1024)
    if verbose:
        full_msg+="\ntorch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024)
        full_msg+="\ntorch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024)
    print(full_msg, flush=True)
    torch.cuda.reset_peak_memory_stats(device='cuda') # reset since we reported it
    #if verbose: print(torch.cuda.memory_summary()) # way too much info
