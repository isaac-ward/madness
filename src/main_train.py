from dotenv import load_dotenv
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

import utils.logging
import dynamics
from learning.data_module import EnvironmentDataModule
from learning.models import PolicyFlowActionDistribution
from agent import Agent
from environment import Environment
import standard

if __name__ == "__main__":

    # Will log everything to here
    log_folder = utils.logging.make_log_folder(name="train")

    # Get the environment variables (api keys)
    load_dotenv()

    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_entity_name = os.getenv("WANDB_ENTITY_NAME")

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
    K = 1000
    H = 50 #int(0.5/dynamics.dt), # X second horizon
    policy = PolicyFlowActionDistribution(
        dynamics=dyn,
        K=K,
        H=H,
        context_input_size=1,
        context_output_size=128,
        num_flow_layers=4,
        learning_rate=1e-3,
    )

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
    wandb_logger = WandbLogger(
        project=os.getenv('WANDB_PROJECT_NAME'),
        entity=os.getenv('WANDB_API_KEY'),
        api_key=os.getenv('WANDB_ENTITY_NAME'),
    )

    # Create a ModelCheckpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        # Monitor the validation loss, and save the best model
        monitor="val/loss",
        mode="min",
        dirpath=f"{log_folder}/checkpoints",
        filename="min_val_loss", 
        save_top_k=1,   # Save only the best model
    )

    # Create a trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=32,
        check_val_every_n_epoch=4,
        progress_bar_refresh_rate=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # Perform the training (and validation)
    trainer.fit(data_module)