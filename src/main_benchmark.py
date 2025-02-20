import numpy as np
from tqdm import tqdm
import scipy
import os
import pickle
import csv
import time
import copy
import matplotlib.pyplot as plt
import cupy as cp

import utils.general
import utils.logging
from environment import Environment
from mapping import Map
from agent import Agent
from visual import Visual
from policies.simple import PolicyNothing, PolicyRandom, PolicyConstant
from policies.mppi import PolicyMPPI
import policies.samplers
import standard
from benchmarking.benchmarker import Benchmarker

# TODO implement wandb to allow for more efficient grid searching of parameters

if __name__ == "__main__":

    # Get the dynamics and maps
    dyn = standard.get_standard_dynamics()

    # Create a map representation
    maps = [
        #standard.get_chamber(),
        standard.get_tunnels(),
        #standard.get_28x28x28_at_111(),
        #standard.get_28x28x28_at_111_with_obstacles(),
    ]

    # So we'll start with the simplest map, and t yellow hen we'll
    # attempt to do all the episodes for a single agent, doing
    # all agents sequentially. So its map (series) * episodes (parallel) * agents (series)
    
    # Loop through all the maps
    for map_ in maps:
        print(f"Running on map: {map_.map_name}")

        # Create the environment
        num_seconds = 30
        num_steps = int(num_seconds / dyn.dt)
        environment = Environment(
            state_initial=np.zeros(dyn.state_size()),
            state_goal=np.zeros(dyn.state_size()),
            dynamics=dyn,
            map_=map_,
            episode_length=num_steps,
        )
    
        # Set it up on this environment (will be repeatedly reset)
        benchmarker = Benchmarker(
            environment=environment,
            num_episodes=1,
            log_folder=utils.logging.make_log_folder(name=f"benchmark-{map_.map_name}", make_handy_subfolders=False),
        )

        # Run the benchmark (parallelization will be used)
        benchmarker.benchmark(
            agent_functions=[
                Benchmarker.get_mppi_agent,
                # Benchmarker.get_alilqr_agent,
                # Benchmarker.get_flowmppi_agent,
                # Benchmarker.get_flowmppi_alilqr_agent,
            ],
            render_videos=True
        )
