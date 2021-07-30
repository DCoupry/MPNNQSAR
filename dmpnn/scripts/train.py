#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
              \     /
          \    o ^ o    /
            \ (     ) /
 ____________(%%%%%%%)____________
(     /   /  )%%%%%%%(  \   \     )
(___/___/__/           \__\___\___)
   (     /  /(%%%%%%%)\  \     )
    (__/___/ (%%%%%%%) \___\__)
            /(       )\
          /   (%%%%%)   \
               (%%%)
                 !
"""
from dmpnn.models import GraphNetworkLigthning, GraphNetworkEnsembleLightning, ExplainerNetworkLightning
from dmpnn.data import GraphsDataModule
from dmpnn.splits import ScaffoldSplitter

import torch
import numpy
import os
import shutil
import optuna
import argparse
import pandas
import warnings
import logging
import sys
import time
import pytorch_lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from typing import Union, Tuple, Optional

__author__ = "Damien Coupry"
__copyright__ = "GSK"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Damien Coupry"
__email__ = "damien.x.coupry@gsk.com"
__status__ = "Dev"

# setting up loggers and seeds
logging.getLogger("lightning").setLevel(logging.ERROR)
logger = logging.getLogger("optuna")
seed_everything(42)

def train_single_model(args: argparse.Namespace, 
                       data: pytorch_lightning.LightningDataModule, 
                       model_dir: str,
                       pruner: Optional[pytorch_lightning.callbacks.EarlyStopping] = None,
                       ) -> pytorch_lightning.LightningModule:
    steps_per_epoch = int(data.get_train_len() / args.batch_size)
    epochs_per_cycle = int(args.lr_cycle_size / steps_per_epoch)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=epochs_per_cycle + 1)
    callbacks = [early_stopping, ]
    if pruner is not None:
        callbacks += [pruner, ]
    model = GraphNetworkLigthning(hparams=args)
    trainer = Trainer.from_argparse_args(args,
                    auto_select_gpus=True,
                    max_epochs=args.max_epochs,
                    min_epochs=(epochs_per_cycle + 1) * 3,
                    gpus=4,
                    num_nodes=1,
                    logger=False,
                    checkpoint_callback=True,
                    callbacks=callbacks,
                    progress_bar_refresh_rate=1 if args.jobs==1 else 0,
                    precision=16,
                    amp_backend="native",
                    amp_level="O2",
                    default_root_dir=model_dir,
                    weights_summary="full",
                    accelerator="dp",
                    )
    trainer.fit(model, data)
    model.freeze()
    return model


def train(args: argparse.Namespace) -> Tuple[pytorch_lightning.LightningModule, float]:
    """
    [summary]

    Parameters
    ----------
    args : argparse.Namespace
        [description]

    Returns
    -------
    float
        [description]
    """
    model_dir = f"{args.study_name}/{args.model_name}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    models = []
    for cv in range(args.cv):
        reps = []
        for rep in range(3):
            data = GraphsDataModule(hparams=args, cv=cv)
            steps_per_epoch = int(data.get_train_len() / args.batch_size)
            epochs_per_cycle = int(args.lr_cycle_size / steps_per_epoch)
            early_stopping = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=epochs_per_cycle + 1)
            model = GraphNetworkLigthning(hparams=args)
            trainer = Trainer.from_argparse_args(args,
                            auto_select_gpus=True,
                            max_epochs=args.max_epochs,
                            # min_steps=args.lr_cycle_size*3,
                            min_epochs=(epochs_per_cycle + 1) * 3,
                            gpus=4,
                            num_nodes=1,
                            logger=False,
                            checkpoint_callback=True,
                            callbacks=[early_stopping, ], 
                            # gradient_clip_val=0.5,
                            # val_check_interval= min(1.0, 500.0 * args.batch_size / len(data)),
                            progress_bar_refresh_rate=1 if args.jobs==1 else 0,
                            precision=16,
                            amp_backend="native",
                            amp_level="O2",
                            default_root_dir=model_dir,
                            weights_summary=None,
                            accelerator="dp",
                            )
            trainer.fit(model, data)
            trained_model = model._net.eval()
            cv_score = early_stopping.best_score.cpu().detach()
            reps.append((trained_model, cv_score))
        # keep top 1 of 3 initializations
        best_rep = sorted(reps, key=lambda k: k[1])[0]
        models.append(best_rep[0])
    ensemble = GraphNetworkEnsembleLightning(models, model._loss, args.target_col)
    ensemble.freeze()
    torch.save(ensemble._net.cpu().eval(), f"{model_dir}/{args.model_name}.ensemble.ckpt")
    data.setup(stage="test")
    test_score = trainer.test(model=ensemble, test_dataloaders=data.test_dataloader(), verbose=False)
    test_score = test_score[0]['test_loss']
    return ensemble, test_score


def objective(trial: optuna.Trial, prog_args: argparse.Namespace) -> float:
    """
    [summary]

    Parameters
    ----------
    trial : optuna.Trial
        [description]
    prog_args : argparse.Namespace
        [description]

    Returns
    -------
    float
        [description]
    """    
    prog_args = vars(prog_args)
    hparams = {
      'model_name': f"trial-{trial.number}",
      'model_index': trial.number,
      'learning_rate': trial.suggest_loguniform("learning_rate", 5e-5, 5e-3),
      'lr_cycle_size' : trial.suggest_int("graph_depth", 100, 1000),
      'batch_size': trial.suggest_int('batch_size', 32 ,256, log=True),
      'beta1':  trial.suggest_loguniform("beta1", 0.5, 0.999), 
      'beta2':  trial.suggest_loguniform("beta2", 0.9, 0.999),
      'gamma':  trial.suggest_uniform("gamma", 0.8, 0.9),
      'dropout': trial.suggest_uniform("dropout", 0.1, 0.5),
      'weight_decay': trial.suggest_loguniform("weight_decay", 0.001, 0.1),
      'hidden_size': trial.suggest_int('hidden_size', 8 , 32, log=True),
      'graph_depth': trial.suggest_int("graph_depth", 1, 3),
      'fc_depth': trial.suggest_int("fc_depth", 1, 3),
      'output_size': len(prog_args["target_col"]),
    }
    hparams.update(prog_args)
    args=argparse.Namespace(**hparams)
    # trinaing a model with these params
    model_dir = f"{args.study_name}/hyperopt/{args.model_name}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # use the biggest cross validation training set
    data = GraphsDataModule(hparams=args, cv=args.cv-1)
    pruner = optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val_loss")
    model = train_single_model(args, data, model_dir, pruner=pruner)
    data.setup(stage="test")
    test_score = trainer.test(model=model, test_dataloaders=data.test_dataloader(), verbose=False)
    test_score = test_score[0]['test_loss']
    return test_score

def setup_study(args: argparse.Namespace) -> optuna.study.Study:
    """
    [summary]

    Parameters
    ----------
    args : argparse.Namespace
        [description]

    Returns
    -------
    optuna.study.Study
        [description]
    """    
    # generate the study folder
    logger.info(f"Setting up study directory: {args.study_name}")
    if not os.path.exists(args.study_name):
        os.mkdir(args.study_name)
    if not os.path.exists(f"{args.study_name}/hyperopt"):
        os.mkdir(f"{args.study_name}/hyperopt")
        
    sys.stderr = open(f'{args.study_name}/errors.log', 'w')
    # save the splits for cross validation
    logger.info("Reading data...")
    if not os.path.isfile(f"{args.study_name}/study.db"):
        data = pandas.read_csv(args.data_file)#.sample(frac=args.overfit)
        data_fileroot =  os.path.basename(args.data_file)
        logger.info("Scaffold-aware CV splitting:")
        splitter = ScaffoldSplitter(data[args.smiles_col], k=args.cv)
        train_splits, test_split = splitter.get_splits()
        logger.info(f"\t-Test: {len(test_split):6} structures.")
        data.iloc[test_split].to_csv(f"{args.study_name}/{data_fileroot}.test")
        for i, split in enumerate(train_splits):
            train_split = split["train"]
            valid_split = split["valid"]
            logger.info(f"\t-Training   #{i}: {len(train_split):6} structures.")
            logger.info(f"\t-Validation #{i}: {len(valid_split):6} structures.")
            data.iloc[train_split].to_csv(f"{args.study_name}/{data_fileroot}.train.{i}")
            data.iloc[valid_split].to_csv(f"{args.study_name}/{data_fileroot}.valid.{i}")
            GraphsDataModule(hparams=args, cv=i).prepare_data()
    else:
        logger.info("Using pre-generated datasets and splits...")
    study_sampler = optuna.samplers.CmaEsSampler(restart_strategy = 'ipop',
                                                 warn_independent_sampling=False)
    study_pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)
    study = optuna.create_study(study_name=args.study_name, 
                                storage=f'sqlite:///{args.study_name}/study.db',
                                load_if_exists=True,
                                direction="minimize", 
                                sampler=study_sampler,
                                pruner=study_pruner)
    return study

# def train_graph_explainer(args):
#     for i, property_name in enumerate(args.target_col):
#         model = ExplainerNetworkLightning(task=args.task, property_index=i)
#         data = GraphsDataModule(hparams=args, cv=0)
#         early_stopping = EarlyStopping(monitor="val_loss", patience=5)
#         trainer = Trainer(gpus=1,
#                         logger=False,
#                         checkpoint_callback=False,
#                         callbacks=[early_stopping,], 
#                         progress_bar_refresh_rate=1,
#                         weights_summary=None)
#         trainer.fit(model, data)
#         trained_model = model._net.eval()
#         trained_model_file = f"{args.study_name}/graph_{property_name}_explainer.ckpt"
#         torch.save(trained_model.cpu(), trained_model_file)
#     return

def main() -> None:
    """
    [summary]
    """    
    logger.info("***************")
    logger.info("** GSK-DMPNN **")
    logger.info("***************")
    parser = argparse.ArgumentParser(prog="GSK-DMPNN", 
                                     description='Automatically trains an optimal DMPNN ensemble model for QSAR purposes.',
                                     epilog="*** End of optimization ***")
    parser.add_argument("--num_trials", default=10, type=int, help="""Number of hyperparameter optimization rounds to execute. 
                                                                      Each trial trains a number of models equal to the square 
                                                                      of the cross validation parameter.""")
    parser.add_argument("--study_name", default="study", type=str, help="The name of the optimization study. A directory will be created")
    parser.add_argument("--cv", default=3, type=int, help="The number of scaffold aware cross-validation folds and repeats.")
    parser.add_argument('--task', choices=["regression", "classification"], type=str, help="What task to train the model on.")
    parser.add_argument('--max_epochs',  default=100, type=int, help="Maximum epochs.") 
    parser.add_argument("--jobs",  type=int, default=1, help="The number of available gpus")
    parser.add_argument("--lr_cycle_size",  default=2000, type=int, help="LR is cycled every N iterations.") 
    parser.add_argument("--extra",  default=None, nargs="+", type=str, help="Extra feaures generators.")    
    # give the module a chance to add own params
    parser = GraphsDataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    study = setup_study(args)
    study.optimize(lambda trial : objective(trial, args), 
                   n_trials=args.num_trials, 
                   timeout=48*60*60, 
                   n_jobs=args.jobs,
                   catch=(RuntimeError, ))
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f'{args.study_name}/importances.svg')
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f'{args.study_name}/history.svg')
    trial = study.best_trial
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial: {trial.number}")
    logger.info(f"\tValue: {trial.value}")
    logger.info("\tParams: ")
    for key, value in trial.params.items():
        logger.info(f"\t\t{key}: {value}")
    print(study.trials_dataframe())
    # best_model_file = f"{args.study_name}/trial-{trial.number}/trial-{trial.number}.ensemble.ckpt"
    # shutil.copyfile(best_model_file, f"{args.study_name}/best_model.ckpt")
    # study.trials_dataframe().to_csv(f"{args.study_name}/study_results.csv")
    # train_graph_explainer(args)
    return None

if __name__ == '__main__':
    # run the code
    main()
