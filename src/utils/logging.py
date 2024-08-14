import os
import numpy as np
import pickle
from utils.general import get_logs_dir, get_timestamp

def make_log_folder(name="run"):
    folder_name = f"{name}_{get_timestamp()}"
    log_folder = os.path.join(get_logs_dir(), folder_name)
    os.makedirs(log_folder)

    # We then need to make the following subfolders
    subfolders = [
        "agent",
        "environment",
        "visuals",
        "policy",
    ]
    for subfolder in subfolders:
        ensure_log_subfolder_exists(log_folder, subfolder)

    return log_folder

def ensure_log_subfolder_exists(log_folder, subfolder_name):
    subfolder = os.path.join(log_folder, subfolder_name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

def pickle_to_filepath(filepath, object):
    """
    Pickle an object to a folder
    """
    with open(filepath, "wb") as f:
        pickle.dump(object, f)

def unpickle_from_filepath(filepath):
    """
    Unpickle an object from a filepath
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)

def save_state_and_action_trajectories(
    folder_save,
    state_trajectories,
    action_trajectories,
    suffix="",
):
    """
    Export the history of the environment to a folder
    as npz files
    """

    if suffix != "": suffix = f"_{suffix}"

    np.savez(os.path.join(folder_save, f"state_trajectories{suffix}.npz"), np.array(state_trajectories))
    np.savez(os.path.join(folder_save, f"action_trajectories{suffix}.npz"), np.array(action_trajectories))

def load_state_and_action_trajectories(
    folder_load,
    suffix="",
):
    """
    Load the state and action trajectories from a folder
    """

    if suffix != "": suffix = f"_{suffix}"

    state_trajectories = np.load(os.path.join(folder_load, f"state_trajectories{suffix}.npz"), allow_pickle=True)["arr_0"]
    action_trajectories = np.load(os.path.join(folder_load, f"action_trajectories{suffix}.npz"), allow_pickle=True)["arr_0"]
    return state_trajectories, action_trajectories