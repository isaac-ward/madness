import os 
import time
import datetime
import hashlib
import pickle
import random
import torch

import numpy as np

def random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def time_based_random_seed():
    # Use down to the microsecond
    seed = int(time.time() * 1e6) % (2**32)
    random_seed(seed)

def get_time_based_rng():
    seed = int(time.time() * 1e6) % (2**32)
    return np.random.default_rng(seed)

def get_timestamp(ultra_precise=False):
    if ultra_precise:
        # Really want uniqueness here, so we go down to the nanosecond
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S") + "-%06d" % now.microsecond
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

def get_dotenv_path():
    return os.path.join(get_project_dir(), 'env/.env')

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

class ItemHistoryTracker:
    """
    This class is useful for representing state and action histories,
    and provides utilities for taking the last n items with zero padding 
    """
    def __init__(self, item_shape):
        self.history = []
        self.item_shape = item_shape

    def append(self, item):
        self.history.append(item)

    def get_last_n_items_with_zero_pad(self, n):
        """
        If we have less than n items, we pad with zeros at the start
        of the returned list
        """
        items_available = len(self.history)
        if items_available >= n:
            return np.array(self.history[-n:])
        else:
            return np.array([np.zeros(self.item_shape) for _ in range(n - items_available)] + self.history)
    
    def get_history(self):
        return np.array(self.history)

    def get_item(index):
        return self.history[index]

    def get_first_item(self):
        return self.history[0]

    def get_last_item(self):
        return self.history[-1]

    def reset(self):
        self.history = []

    def __len__(self):
        return len(self.history)