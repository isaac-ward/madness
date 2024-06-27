import numpy as np
from tqdm import tqdm
import scipy
import os
import pickle
import csv
import time

import dynamics
import utils.general

if __name__ == "__main__":

    # Will log everything to here
    log_folder = utils.general.make_log_folder(name="run")

    