#!/usr/bin/env python
# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.

Usage:

.. code-block:: console

    python examples/example_entrypoint.py -f <path_to_yaml> --algorithms label_smoothing --alpha 0.1
"""

import logging
import os
import sys
import tempfile
import warnings

from composer.loggers import LogLevel
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import dist

import experimental


def _warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def main():

    global_rank = dist.get_global_rank()

    logging.basicConfig(
        # Example of format string
        # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: composer.trainer.trainer: Using precision Precision.FP32
        # Including the PID and thread name to help with debugging dataloader workers and callbacks that spawn background
        # threads / processes
        format=f'%(asctime)s: rank{global_rank}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s')

    # register all algorithms
    experimental.register_all_algorithms()

    warnings.formatwarning = _warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv.append('--help')

    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    trainer = hparams.initialize_object()

    # if using wandb, store the config inside the wandb run
    try:
        import wandb
    except ImportError:
        pass
    else:
        if wandb.run is not None:
            wandb.config.update(hparams.to_dict())

    # Only log the config once, since it should be the same on all ranks.
    if global_rank == 0:
        save_and_upload_config(hparams, trainer)

    # Print the config to the terminal
    if dist.get_local_rank() == 0:
        print('*' * 30)
        print('Config:')
        print(hparams.to_yaml())
        print('*' * 30)

    trainer.fit()


def save_and_upload_config(hparams, trainer):
    """Saves the provided hparams file and uploads to any object stores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hparams_name = os.path.join(tmpdir, 'hparams.yaml')
        with open(hparams_name, 'w+') as f:
            f.write(hparams.to_yaml())
        trainer.logger.file_artifact(
            LogLevel.FIT,
            artifact_name=f'{trainer.state.run_name}/hparams.yaml',
            file_path=f.name,
            overwrite=True,
        )


if __name__ == '__main__':
    main()
