import logging
import os
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from itertools import combinations
import torch

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    def run_training_with_transformations(config):
        """
        Run the training loop and return the loss for a given configuration of transformations.
        """
        # Instantiate the model, data module, and trainer using the current config
        module = instantiate(config.module, optimizer=config.optimizer, lr_scheduler=config.lr_scheduler)
        datamodule = instantiate(config.datamodule,
                                train_transform=_build_transform(config.transforms.train),
                                val_transform=_build_transform(config.transforms.val),
                                test_transform=_build_transform(config.transforms.test))

        trainer = pl.Trainer(**config.trainer)
        
        # Fit the model
        trainer.fit(module, datamodule)
        
        # Return validation loss or another relevant metric
        val_metrics = trainer.validate(module, datamodule)
        return val_metrics[0]['loss']  # Or another metric you're interested in

    def grid_search_transformations(transforms_list, config):
        best_loss = float('inf')
        best_transform_combination = None

        # Generate all combinations of transformations
        for num_transforms in range(1, len(transforms_list) + 1):
            for transform_comb in combinations(transforms_list, num_transforms):
                # Create a new config with the current combination of transformations
                config.transforms.train = [instantiate(cfg) for cfg in transform_comb]

                # Run training with this combination of transformations
                loss = run_training_with_transformations(config)

                # Track the best performing combination
                if loss < best_loss:
                    best_loss = loss
                    best_transform_combination = transform_comb

        return best_transform_combination, best_loss

    # List of transformations to search over
    transforms_list = config.transforms.search

    # Perform grid search to find the best combination of transformations
    best_transform_combination, best_loss = grid_search_transformations(transforms_list, config)

    log.info(f"Best Transform Combination: {best_transform_combination}")
    log.info(f"Best Loss: {best_loss}")

    # Update config with the best combination of transformations
    config.transforms.train = [instantiate(cfg) for cfg in best_transform_combination]

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.module}")
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        module = module.load_from_checkpoint(
            config.checkpoint,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            decoder=config.decoder,
        )

    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    # Initialize trainer
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
    )

    if config.train:
        # Check if a past checkpoint exists to resume training from
        checkpoint_dir = Path.cwd().joinpath("checkpoints")
        resume_from_checkpoint = utils.get_last_checkpoint(checkpoint_dir)
        if resume_from_checkpoint is not None:
            log.info(f"Resuming training from checkpoint {resume_from_checkpoint}")

        # Train
        trainer.fit(module, datamodule, ckpt_path=resume_from_checkpoint)

        # Load best checkpoint
        module = module.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Validate and test on the best checkpoint (if training), or on the
    # loaded `config.checkpoint` (otherwise)
    val_metrics = trainer.validate(module, datamodule)
    test_metrics = trainer.test(module, datamodule)

    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": trainer.checkpoint_callback.best_model_path,
    }
    pprint.pprint(results, sort_dicts=False)

if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()