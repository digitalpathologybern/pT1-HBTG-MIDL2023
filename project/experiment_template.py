import logging
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import json

import pytorch_lightning as pl
import torch_geometric
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torchmetrics import MetricCollection, Accuracy
import wandb

import util.analytics as analytics
from util.custom_transforms import AddPosToNodeFeature


class Experiment:
    def __init__(self, multi_run: int, output_folder: str, cross_val: int = None, rebuild_dataset: bool = False,
                 split_json: str = None, balanced_loading_train_val: bool = False, **kwargs):
        self.kwargs = kwargs
        self.multi_run = multi_run
        self.split_json_path = split_json
        self.split_json = split_json
        self.cross_val = cross_val
        self.experiment_id = f'{self.kwargs["experiment_name"]}-{datetime.now():%Y-%m-%d_%H-%M-%S}'
        self.rebuild_dataset = rebuild_dataset
        self.balanced_loading_train_val = balanced_loading_train_val

        # Set visible GPUs ensure that only one GPU is used (parallelize?)
        self.kwargs['gpu_id'] = self.gpu_ind = self.get_gpu_id(**self.kwargs)

        # create the output folder
        self.kwargs['output_folder'] = self.setup_output_folder(output_folder=output_folder)

        # TODO: change this so that seed is different for every run?
        if kwargs['seed']:
            pl.seed_everything(kwargs['seed'])

        # should we load the saved dataset from processed or should it be rebuilt
        input_folder = self.kwargs['input_folder']
        processed_path = os.path.join(input_folder, 'processed')
        if os.path.exists(processed_path):
            if self.rebuild_dataset:
                logging.warning(
                    f'Datasets will be rebuilt! Deleting folder {input_folder}/processed and processing dataset fresh.')
                shutil.rmtree(processed_path)
            else:
                logging.warning('Datasets is NOT rebuilt!')
        # set datasets
        self.train_ds, self.val_ds, self.test_ds = self.get_datasets()

        # set metrics
        self.set_metrics()

    # ------------
    # properties
    # ------------
    @property
    def cross_val(self):
        return self._cross_val

    @cross_val.setter
    def cross_val(self, cross_val: str):
        if cross_val is None and self.split_json is None:
            cv = None
        else:
            split_l = list(self.split_json.keys())
            if split_l[0].isdigit():
                # we have a list of cv splits
                cv = len([int(i) for i in self.split_json.keys()])
            else:
                # we have train / val / test
                cv = None
        self._cross_val = cv

    @property
    def split_json(self):
        return self._split_json

    @split_json.setter
    def split_json(self, split_json: str):
        if split_json is not None:
            with open(split_json) as f:
                data_split = json.load(f)
        else:
            data_split = None
        self._split_json = data_split

    @property
    def class_frequencies(self):
        # calculate the class frequency on the training set
        class_frequencies = analytics.get_class_freq(self.train_ds)
        return class_frequencies

    @property
    def num_features(self):
        nb = self.train_ds.num_features
        if sum([isinstance(i, AddPosToNodeFeature) for i in self.train_ds.transform.transforms]) > 0:
            nb += 2
        return nb

    @property
    def num_edge_features(self):
        nb = self.train_ds.num_edge_features
        return nb

    @property
    def num_classes(self):
        if isinstance(self.train_ds, list):
            return self.train_ds[0].num_classes
        else:
            return self.train_ds.num_classes

    @property
    def accelerator(self):
        accelerator = 'gpu' if self.gpu_ind is not None else 'cpu'
        if self.kwargs['no_cuda']:
            accelerator = 'cpu'
        return accelerator

    # ------------
    # metrics
    # ------------
    def set_metrics(self):
        self.train_metric = Accuracy()
        self.val_metric = Accuracy()
        self.test_metric = Accuracy()

    # ------------
    # data
    # ------------
    def get_datasets(self, balance_loading_train_val: bool = False):
        if self.split_json is not None:
            self.ds = self.get_dataset_full()
            split_list = np.array(self.ds.data.subset)
            # specified split is train / val / test
            if 'train' in split_list:
                train_ds, val_ds, test_ds = self.ds[split_list == 'train'], self.ds[split_list == 'val'], self.ds[
                    split_list == 'test']
                # TODO: make this dynamic?
                if self.balanced_loading_train_val:
                    train_ds.upsample()
                    val_ds.upsample()
            # specified splits are CV splits
            else:
                self.cv_folds = np.unique(split_list)
                train_ds, val_ds, test_ds = self.update_cv_splits(0, return_ds=True)
                return train_ds, val_ds, test_ds

        else:
            return self.get_dataset_splits()

    def update_cv_splits(self, cv_ind, return_ds=False):
        split_list = np.array(self.ds.data.subset)
        test_mask = split_list == self.cv_folds[cv_ind]
        val_mask = split_list == self.cv_folds[(cv_ind + 1) % len(self.cv_folds)]
        train_mask = ~(test_mask | val_mask)
        train_ds, val_ds, test_ds = self.ds[train_mask], self.ds[val_mask], self.ds[test_mask]
        if return_ds:
            return train_ds, val_ds, test_ds
        else:
            self.train_ds = train_ds
            self.val_ds = val_ds
            self.test_ds = test_ds

    def get_dataset_splits(self):
        return NotImplementedError

    def get_dataset_full(self):
        return NotImplementedError

    def get_dataloaders(self):
        # Set up the transforms
        self.train_ds.transform = self.get_train_transforms()
        self.val_ds.transform = self.get_val_transforms()
        self.test_ds.transform = self.get_train_transforms()
        train_loader, val_loader, test_loader = self.get_torch_dataloaders(**self.kwargs)
        return train_loader, val_loader, test_loader

    def get_train_transforms(self):
        return NotImplementedError

    def get_val_transforms(self):
        return self.get_train_transforms()

    def get_test_transforms(self):
        return self.get_train_transforms()

    @staticmethod
    def get_pre_transforms():
        return NotImplementedError

    def get_torch_dataloaders(self, batch_size, workers, **kwargs) -> tuple:
        # Setup dataloaders
        logging.debug('Setting up dataloaders')

        train_loader = torch_geometric.loader.DataLoader(self.train_ds,
                                                         shuffle=True,
                                                         batch_size=batch_size,
                                                         num_workers=workers)
        val_loader = torch_geometric.loader.DataLoader(self.val_ds,
                                                       batch_size=batch_size,
                                                       num_workers=workers)
        test_loader = torch_geometric.loader.DataLoader(self.test_ds,
                                                        batch_size=batch_size,
                                                        num_workers=workers)
        return train_loader, val_loader, test_loader

    # ------------
    # model
    # ------------
    def get_model(self, run_id=None, cv_ind=None):
        return NotImplementedError

    def get_checkpoint_callback(self, run_id=None, cv_ind=None) -> pl.callbacks.ModelCheckpoint:
        # default is last model is saved
        run_tag = '' if run_id is not None else f'_run{run_id}'
        cv_tag = '' if cv_ind is not None else f'_cv{cv_ind}'
        checkpoint_filename = f'{self.kwargs["experiment_name"]}{run_tag}{cv_tag}' + '_{epoch:02d}'
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=self.kwargs['output_folder'],
                                                           filename=checkpoint_filename)
        return checkpoint_callback

    # ------------
    # run the experiments
    # ------------
    def one_run(self, train_loader, val_loader, test_loader, run_id=None, cv_ind=None):
        # ------------
        # training
        # ------------
        # checkpoint requirements
        checkpoint_callback = self.get_checkpoint_callback(run_id=run_id, cv_ind=cv_ind)

        trainer = pl.Trainer(gpus=self.gpu_ind, max_epochs=self.kwargs['epochs'], log_every_n_steps=1,
                             logger=self.loggers, callbacks=[checkpoint_callback], accelerator=self.accelerator)

        model = self.get_model(run_id=run_id, cv_ind=cv_ind)
        # log hparams to wandb
        for logger in self.loggers:
            if hasattr(logger.experiment, 'config'):
                logger.experiment.config.update(model.hparams)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # ------------
        # testing
        # ------------
        result = trainer.test(dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)
        # prepare softmax predictions for saving
        softmax_df = self.format_softmax(model.softmax_output, run_id=run_id)

        return result, softmax_df

    def main_run(self, train_loader, val_loader, test_loader, cv_ind=None):
        test_summary = pd.DataFrame()
        all_softmax_df = pd.DataFrame()
        nb_params = sum(p.numel() for p in self.get_model().parameters() if p.requires_grad)

        # without multi-run
        if self.multi_run is None or self.multi_run < 2:
            if cv_ind is None:
                experiment_name = self.kwargs["experiment_name"]
                self.loggers = self.get_loggers(experiment_name)
                series_name = None
            else:
                experiment_name = f'{self.kwargs["experiment_name"]}-cv{cv_ind}'
                self.loggers = self.get_loggers(experiment_name)
                series_name = f'cv{cv_ind}'

            metrics, softmax_df = self.one_run(train_loader, val_loader, test_loader, cv_ind=cv_ind)
            # update metrics summary
            test_summary = pd.concat([test_summary, pd.Series(metrics[0], name=series_name)], axis=1)
            all_softmax_df = pd.concat([all_softmax_df, softmax_df], axis=1)
            for logger in self.loggers:
                if hasattr(logger.experiment, 'config'):
                    logger.experiment.config.update({'nb_trainable_params': nb_params})

        else:
            # with multi-run
            for run in range(self.multi_run):
                if cv_ind is None:
                    experiment_name = f'{self.kwargs["experiment_name"]}'
                    self.loggers = self.get_loggers(f'{experiment_name}-run{run}')
                    os.environ["WANDB_RUN_GROUP"] = f'run-{run}'
                    metrics, softmax_df = self.one_run(train_loader, val_loader, test_loader, run_id=run)
                else:
                    experiment_name = f'{self.kwargs["experiment_name"]}-cv{cv_ind}'
                    self.loggers = self.get_loggers(f'{experiment_name}-run{run}')
                    os.environ["WANDB_RUN_GROUP"] = f'run-{run}'
                    metrics, softmax_df = self.one_run(train_loader, val_loader, test_loader, run_id=run, cv_ind=cv_ind)
                # update metrics summary
                test_summary = pd.concat([test_summary, pd.Series(metrics[0], name=f'run{run}')], axis=1)
                all_softmax_df = pd.concat([all_softmax_df, softmax_df], axis=1)
                # logger clean-up
                for logger in self.loggers:
                    if logger.__class__ is pl.loggers.WandbLogger:
                        logger.experiment.finish()

            # calculate the averages over all the runs
            test_mean = test_summary.mean(axis=1)
            test_name = 'mean' if cv_ind is None else f'cv{cv_ind}'
            test_mean.rename(test_name, inplace=True)
            # test_summary_avg = pd.concat([test_summary.mean(axis=1), test_summary.std(axis=1)], axis=1, keys=['mean', 'std'])
            # print(f'Average on the test set over {self.multi_run} runs:')
            # for metric in test_performances.keys():
            #     print(f'{metric}: {summary_dict[f"{metric}_mean"]:.2f} (+- {summary_dict[f"{metric}_sd"]:.2f})')

        return test_mean, all_softmax_df

    def main(self):
        """
        Main method, runs the experiments
        """
        test_summary = pd.DataFrame()
        all_softmax_df = pd.DataFrame()
        # log the averages
        # with cross validation
        if self.cross_val:
            for cv_ind in range(self.cross_val):
                if cv_ind > 0:
                    # reload the datasets
                    self.update_cv_splits(cv_ind)
                train_loader, val_loader, test_loader = self.get_dataloaders()
                r_df, softmax_df = self.main_run(train_loader, val_loader, test_loader, cv_ind)
                # update metrics summary
                test_summary = pd.concat([test_summary, r_df], axis=1)
                softmax_df.insert(0, 'cv-fold', [cv_ind] * len(softmax_df.index))
                all_softmax_df = pd.concat([all_softmax_df, softmax_df])

                for logger in self.loggers:
                    if logger.__class__ is pl.loggers.WandbLogger:
                        logger.experiment.finish()

            test_summary_avg = pd.concat([test_summary.mean(axis=1), test_summary.std(axis=1)], axis=1,
                                         keys=['mean', 'std'])

        # without cross validation
        else:
            train_loader, val_loader, test_loader = self.get_dataloaders()
            test_summary_avg, all_softmax_df = self.main_run(train_loader, val_loader, test_loader)

        nb_params = sum(p.numel() for p in self.get_model().parameters() if p.requires_grad)
        test_summary_avg['nb_params'] = [nb_params] * len(test_summary_avg.index)

        summary_path = self.kwargs["output_folder"] / 'files' / f'{self.kwargs["experiment_name"]}-summary.csv'
        summary_df = pd.concat([test_summary, test_summary_avg], axis=1)
        summary_df.to_csv(summary_path)

        # save all_softmax_df
        csv_softmax_path = self.kwargs["output_folder"] / 'files' / f'{self.kwargs["experiment_name"]}-softmax.csv'
        all_softmax_df = all_softmax_df.reset_index(level=1)
        all_softmax_df.to_csv(csv_softmax_path)
        # wandb.save(str(csv_softmax_path))

        # # log averages
        avg_logger = WandbLogger(name=f'{self.kwargs["experiment_name"]}',
                                 project=self.kwargs['wandb_project'],
                                 save_dir=f'{self.kwargs["output_folder"]}',
                                 group='average-overview',
                                 reinit=True,
                                 config={})
        avg_logger.experiment.log(test_summary_avg.to_dict())
        avg_logger.experiment.finish()

        print('All done, yay :)!')

    @staticmethod
    def get_gpu_id(gpu_id: list, no_cuda: bool, **kwargs):
        # Set visible GPUs ensure that only one GPU is used (parallelize?)
        ind = [0]
        if no_cuda:
            ind = None
        else:
            if gpu_id is not None:
                if len(gpu_id) > 1:
                    logging.warning(f"This runner can only be used on one GPU at a time, selecting GPU {ind}")
                ind = gpu_id
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(ind)
        return ind

    def setup_output_folder(self, output_folder):
        # create the output folder
        output_path = Path(output_folder) / self.experiment_id
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def format_softmax(self, softmax_output, run_id=None):
        file_ids_test = [i.split('.gxl')[0] for i in self.test_ds.config['file_names']]
        softmax_pre = softmax_output.compute().cpu().numpy()
        run_id = f'r{run_id}_' if run_id is not None else ""
        softmax_d = {f'{run_id}softmax_class{i}': column for i, column in enumerate(softmax_pre[:, 2:].T)}
        softmax_df = pd.DataFrame.from_dict(softmax_d)
        multi_ind = pd.MultiIndex.from_tuples([(file_id, label) for file_id, label
                                               in zip(np.array(file_ids_test)[softmax_pre[:, 0].astype(int)],
                                                      softmax_pre[:, 1].astype(int))],
                                              names=('file_id', 'class_label'))
        softmax_df.set_index(multi_ind, inplace=True)
        return softmax_df

    def get_loggers(self, exp_name, **kwargs):
        # logging
        loggers = []

        csv_logger = CSVLogger(save_dir=f'{self.kwargs["output_folder"]}', name=exp_name)
        loggers.append(csv_logger)
        # with wandb
        if self.kwargs['wandb_project']:
            wandb_logger = WandbLogger(name=exp_name,
                                       project=self.kwargs['wandb_project'],
                                       save_dir=f'{self.kwargs["output_folder"]}',
                                       group=self.experiment_id,
                                       reinit=True,
                                       config={})
            loggers.append(wandb_logger)
        return loggers
