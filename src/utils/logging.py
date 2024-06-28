import os
import numpy as np
from utils.general import get_logs_dir, get_timestamp

def make_log_folder(name="run"):
    folder_name = f"{name}_{get_timestamp()}"
    log_folder = os.path.join(get_logs_dir(), folder_name)
    os.makedirs(log_folder)
    return log_folder

def log_histories(
    folder_save,
    state_history,
    action_history,
):
    """
    Export the history of the environment to a folder
    as npz files
    """
    # Create a folder to save in if it does not already exist
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)

    # Save the state and action history to here
    np.savez(os.path.join(folder_save, "state_history.npz"), np.array(state_history))
    np.savez(os.path.join(folder_save, "action_history.npz"), np.array(action_history))