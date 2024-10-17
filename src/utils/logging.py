import os
import numpy as np
import pickle
import warnings
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

def pickle_to_filepath(filepath, object, verbose=False):
    """
    Pickle an object to a folder
    """

    # Warn if the file already exists
    if os.path.exists(filepath) and verbose:
        warnings.warn(f"File already exists at {filepath}. Overwriting.")

    with open(filepath, "wb") as f:
        pickle.dump(object, f)

def save_to_npz(filepath, array, verbose=False):
    """
    Save an array to a npz file
    """

    # Ensure the folderpath exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Warn if the file already exists
    if os.path.exists(filepath) and verbose:
        warnings.warn(f"File already exists at {filepath}. Overwriting.")

    np.savez(filepath, array)

def load_from_npz(filepath):
    """
    Load an array from a npz file
    """
    return np.load(filepath, allow_pickle=True)["arr_0"]

def unpickle_from_filepath(filepath):
    """
    Unpickle an object from a filepath
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)

def write_shape_to_text_file(filepath, array):
    """
    Write the shape of an array to a text file
    """
    with open(filepath, "w") as f:
        f.write(str(array.shape))

def write_preview_to_text_file(filepath, array, num_entries=4):
    """
    Write a preview of an array to a text file
    """
    with open(filepath, "w") as f:
        f.write(str(array[:num_entries]))
        f.write("\n...\n")
        f.write(str(array[-num_entries:]))

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

    # For debugging purposes save the shapes to text files
    write_shape_to_text_file(os.path.join(folder_save, f"state_trajectories_shape{suffix}.txt"), state_trajectories)
    write_shape_to_text_file(os.path.join(folder_save, f"action_trajectories_shape{suffix}.txt"), action_trajectories)

    # For debugging purposes save the first 4 and last 4 entries to text files
    write_preview_to_text_file(os.path.join(folder_save, f"state_trajectories_preview{suffix}.txt"), state_trajectories)
    write_preview_to_text_file(os.path.join(folder_save, f"action_trajectories_preview{suffix}.txt"), action_trajectories)

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