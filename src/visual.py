import matplotlib as mpl 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import os
import moviepy.editor as mpy
from tqdm import tqdm
import math
from concurrent.futures import ProcessPoolExecutor

# Given a run folder, generate visual assets