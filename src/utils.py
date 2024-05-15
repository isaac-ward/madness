import os 
import datetime

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

def make_log_folder(name="run"):
    folder_name = f"{name}_{get_timestamp()}"
    log_folder = os.path.join(get_logs_dir(), folder_name)
    os.makedirs(log_folder)
    return log_folder