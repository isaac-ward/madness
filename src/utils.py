import os 
import datetime
import hashlib

def get_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
    return timestamp

def get_project_dir():
    # We're currently in madness/src/utils.py
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_assets_dir():
    return os.path.join(get_project_dir(), 'assets')

def get_logs_dir():
    return os.path.join(get_project_dir(), 'logs')

def get_cache_dir():
    return os.path.join(get_project_dir(), 'cache')

def make_log_folder(name="run"):
    folder_name = f"{name}_{get_timestamp()}"
    log_folder = os.path.join(get_logs_dir(), folder_name)
    os.makedirs(log_folder)
    return log_folder

def compute_hash(*args):
    hash_obj = hashlib.md5()
    for arg in args:
        hash_obj.update(repr(arg).encode())
    return hash_obj.hexdigest()