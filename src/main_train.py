from dotenv import load_dotenv
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchinfo import summary

import utils.general
import utils.logging
from utils.learning import make_k1_checkpoint_callback
import dynamics
from learning.data_module import EnvironmentDataModule
from learning.models import PolicyFlowActionDistribution
from agent import Agent
from environment import Environment
import standard

torch.cuda.empty_cache()
# Set the precision of the matrix multiplication to trade off precision 
# for performance
# 'medium' | 'high'
torch.set_float32_matmul_precision('high') 

if __name__ == "__main__":

    # Will log everything to here
    log_folder = utils.logging.make_log_folder(name="train")

    # Get the environment variables (api keys)
    load_dotenv(dotenv_path=utils.general.get_dotenv_path())

    # Create the standard objects needed for this paradigm
    dyn = standard.get_standard_dynamics_quadcopter_3d()
    map_ = standard.get_standard_map()

    # Create the environment - the state_initial and state_goal 
    # are randomly generated in the dataset, so we don't need to 
    # specify them here
    num_seconds = 16
    num_steps = int(num_seconds / dyn.dt)
    environment = Environment(
        state_initial=None,
        state_goal=None,
        dynamics=dyn,
        map_=map_,
        episode_length=num_steps,
    )

    # Create the agent, which has an initial state and a policy
    K = 7
    H = 50 #int(0.5/dynamics.dt), # X second horizon
    policy = PolicyFlowActionDistribution(
        dynamics=dyn,
        K=K,
        H=H,
        lambda_=None,
        map_=map_,
        context_input_size=2*dyn.state_size(),
        context_output_size=64,
        num_flow_layers=3,
        learning_rate=1e-3,
    )
    summary(policy)

    # Can now create an agent, the state_initial will be set by the
    # dataset during training
    agent = Agent(
        state_initial=None,
        policy=policy,
        state_size=dyn.state_size(),
        action_size=dyn.action_size(),
    ) 

    # Create a data module, and a trainer, and get to learnin'!
    data_module = EnvironmentDataModule(
        environment=environment,
        agent=agent,
        batch_size=1,
    )

    # We'll log with wandb
    os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY')
    wandb_logger = WandbLogger(
        project=os.getenv('WANDB_PROJECT_NAME'),
        entity=os.getenv('WANDB_ENTITY_NAME'),
    )

    # Create a trainer
    trainer = pl.Trainer(
        max_epochs=32,
        check_val_every_n_epoch=4,
        # Change hardware settings accordingly
        devices=[1],
        accelerator="gpu",
        #progress_bar_refresh_rate=1,
        logger=wandb_logger,
        callbacks=[
            make_k1_checkpoint_callback(log_folder, "val/reward/best", "max"),
            make_k1_checkpoint_callback(log_folder, "val/reward/mean", "max"),
            make_k1_checkpoint_callback(log_folder, "val/loss", "min"),
        ],
    )

    # Perform the training (and validation)
    trainer.fit(policy, data_module)