import os 
import datetime
import hashlib
import pickle

import numpy as np

def seed(seed):
    np.random.seed(seed)

def get_timestamp(ultra_precise=False):
    if ultra_precise:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
    return timestamp

def get_project_dir():
    # We're currently in madness/src/utils/logging.py
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_assets_dir():
    return os.path.join(get_project_dir(), 'assets')

def get_logs_dir():
    return os.path.join(get_project_dir(), 'logs')

def get_cache_dir():
    return os.path.join(get_project_dir(), 'cache')

def compute_hash(*args):
    hash_obj = hashlib.md5()
    for arg in args:
        hash_obj.update(repr(arg).encode())
    return hash_obj.hexdigest()

class Cacher:
    """
    This class is a time saver. It caches the results of a computation to disk
    based on the hash of the input arguments. If the cache exists, it will load
    the outputs instead of recomputing them. If the cache does not exist, it will
    save the outputs to disk for future use

    Typical usage:

    cacher = Cacher(computation_inputs)
    if cacher.exists():
        outputs = cacher.load()
    else:
        outputs = compute_outputs(computation_inputs)
        cacher.save(outputs)
    """
    def __init__(self, computation_inputs):
        # Hash everything and construct the filepath from the hash
        self.cache_hash = compute_hash(*computation_inputs)
        self.cache_filepath = f"{get_cache_dir()}/cache_{self.cache_hash}.pkl"

    def exists(self):
        return os.path.exists(self.cache_filepath)
    
    def load(self):
        # Assumes that existence has been checked for
        print(f"Loading cache from {self.cache_filepath}")
        with open(self.cache_filepath, "rb") as f:
            return pickle.load(f)
    
    def save(self, outputs_dict):
        with open(self.cache_filepath, "wb") as f:
            pickle.dump(outputs_dict, f)