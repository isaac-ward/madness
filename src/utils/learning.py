from pytorch_lightning.callbacks import ModelCheckpoint

def make_k1_checkpoint_callback(log_folder, monitor, mode):
    return ModelCheckpoint(
        # Monitor some metric and save the best one of those
        monitor=monitor,
        mode=mode,
        dirpath=f"{log_folder}/checkpoints",
        filename=f"{mode}_{monitor.replace('/', '_')}", 
        save_top_k=1,   # Save only the best model
    )
