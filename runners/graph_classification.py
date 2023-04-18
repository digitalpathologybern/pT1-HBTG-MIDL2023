import seaborn as sn
import wandb
from PIL import Image
import pandas as pd
import io
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from torchmetrics import MetricCollection, ConfusionMatrix, Metric, CatMetric

from torch.nn import Module
from torch.nn.functional import softmax
import torch

from util.evaluation import pretty_print_performance


class GraphClassification(pl.LightningModule):
    def __init__(self, model: Module, input_folder: str, output_folder: str, class_frequency: list, num_classes: int,
                 train_metric: MetricCollection, val_metric: MetricCollection, test_metric: MetricCollection,
                 batch_size: int, smoothing: float = 0.2, compute_conf_test: bool = False,
                 compute_conf_val: bool = False,
                 gpu_id: list = None, no_cuda: bool = False, disable_databalancing: bool = False, run_id: str = '',
                 cv_ind: int = None, **kwargs):
        """
        Lightining Module that runs a graph classification task

        :param model: torch Module
            the model that is going to be trained
        :param input_folder: str
            path to the dataset folder
        :param output_folder: str
            path where the results are saved to
        :param class_frequency: list
            frequency of the classes, used compute the weights for the loss function (if disable_databalancing = False)
        :param num_classes: int
            number of classes in the task
        :param gpu_id: int (default 1)
            id of the gpu that the experiments are run on
        :param train_metric: pytorch lightning Metric (default is Accuracy)
            metric(s) that are computed during the training/validation/testing
        :param val_metric: pytorch lightning Metric (default is Accuracy)
            metric(s) that are computed during the training/validation/testing
        :param test_metric: pytorch lightning Metric (default is Accuracy)
            metric(s) that are computed during the training/validation/testing
        :param no_cuda: bool (default False)
            if set, the experiments are run on CPU
        :param disable_databalancing: bool (default False)
            if set, the class weights are not passed to the loss function
        :param run_id: str (default '')
            used in mutli-run to identify the run
        :param compute_conf_val: bool (default False)
            if set, the confusion matrix is also saved every 5 epochs during validation
        :param kwargs: dict
            any additional arguments
        """
        super().__init__()

        # saving the hyper-parameters
        # generic self.save_hyperparameters() does not work with custom architectures
        self.save_hyperparameters('input_folder', 'output_folder', 'class_frequency', 'num_classes',
                                  'disable_databalancing', 'batch_size', 'smoothing')
        kwags_hparam = {name: kwargs[name] for name in
                        ['experiment_name', 'seed', 'epochs', 'ignore_coordinates',
                         'learning_rate', 'step_size', 'momentum', 'weight_decay', 'transforms', 'pre_transforms',
                         'transforms', 'lr_scheduler']
                        if name in kwargs}

        additional_hparams = {**model.hparam, **kwags_hparam, 'optimizer': 'AdamW'}
        for k, v in additional_hparams.items():
            self.hparams[k] = v

        self.model = model

        self.disable_databalancing = disable_databalancing
        self.class_frequency = class_frequency
        self.num_classes = int(num_classes)
        self.gpu_id = gpu_id
        self.no_cuda = no_cuda
        self.run_id = run_id
        self.cv_ind = '' if cv_ind is None else cv_ind
        self.batch_size = batch_size
        self.smoothing = smoothing

        # paths
        self.input_folder = input_folder
        self.output_path = output_folder
        self.output_path_analysis = self.output_path / 'files'
        self.output_path_analysis.mkdir(parents=True, exist_ok=True)

        # metrics
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.compute_conf_val = compute_conf_val
        self.compute_conf_test = compute_conf_test
        self.conf_mat_val = ConfusionMatrix(num_classes=self.num_classes)
        self.conf_mat_test = ConfusionMatrix(num_classes=self.num_classes)
        # collect the softmax output during testing
        self.softmax_output = CatMetric()

    @property
    def loss(self):
        if self.disable_databalancing:
            return torch.nn.CrossEntropyLoss(label_smoothing=self.smoothing)
        else:
            # calculate the weights based on the class frequencies
            # weight = n_samples / (n_classes * np.bincount(y)) or 1/(class_freq*nb_classes)
            weights = torch.from_numpy(1 / (self.class_frequency * self.num_classes))
            # Transfer model to GPU
            if not self.no_cuda:
                # TODO: clean up to gpu moving, should be done differently with pytorch lightning
                weights = weights.to(torch.device(f'cuda:{self.gpu_id[0]}'))
            return torch.nn.CrossEntropyLoss(weight=weights.float(), label_smoothing=self.smoothing)

    def forward(self, batch):
        data = batch
        target = data.y.type(torch.long)  # type cast ist ugly fix for COLORS-3, which loads the classes as floats
        y_hat = self.model(data, batch_size=self.batch_size)
        if len(target) != len(y_hat):  # last batch can have less target values than the batch size
            y_hat = y_hat[:len(target)]
        loss = self.loss(y_hat, target)
        return loss, y_hat

    # TRAINING
    def training_step(self, batch, batch_idx, **kwargs):
        loss, y_hat = self.forward(batch)
        performance = self.train_metric(y_hat, batch.y)
        # log step metric
        self.log(f'train/loss', loss, on_epoch=True, on_step=False, batch_size=self.batch_size)
        if type(performance) == dict:
            performance = pretty_print_performance(performance)
            for metric, val in performance.items():
                self.log(f'train/{metric}', val, on_epoch=True, on_step=False, batch_size=self.batch_size)
        else:
            self.log(f'train/{self.train_metric._get_name()}', performance, on_epoch=True, on_step=False,
                     batch_size=self.batch_size)
        return loss

    # VALIDATION
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.conf_mat_val.reset()

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.forward(batch)
        performance = self.val_metric(y_hat, batch.y)
        self.log(f'val/loss', loss, on_epoch=True, on_step=False, batch_size=self.batch_size)
        if type(performance) == dict:
            performance = pretty_print_performance(performance)
            for metric, val in performance.items():
                self.log(f'val/{metric}', val, on_epoch=True, on_step=False, batch_size=self.batch_size)
        else:
            self.log(f'val/{self.train_metric._get_name()}', performance, on_epoch=True, on_step=False,
                     batch_size=self.batch_size)

        # update the conf mat
        self.conf_mat_val.update(y_hat, batch.y)
        return {f'val/loss': loss, f'val/{self.val_metric._get_name()}': performance}

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        # save the conf matrix every 10 epochs
        if self.current_epoch % 10 == 0 and self.compute_conf_val:
            self._create_and_save_conf_mat(self.conf_mat_val.compute().cpu().numpy(), f'val_{self.current_epoch}')

    # TESTING
    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.conf_mat_test.reset()
        self.softmax_output.reset()

    def test_step(self, batch, batch_idx):
        loss, y_hat = self.forward(batch)
        print(self.model.training)
        # TODO: add temperature scaling
        pred = softmax(y_hat, dim=1)
        # self.test_acc.update(y_hat, batch.y)
        test_output = self.test_metric(y_hat, batch.y)
        self.log(f'test/loss', loss, on_epoch=True, batch_size=self.batch_size)
        if isinstance(test_output, dict):
            test_output = pretty_print_performance(test_output)
            for metric, val in test_output.items():
                self.log(f'test/{metric}', val, on_epoch=True, on_step=False, batch_size=self.batch_size)
        else:
            self.log(f'test/{self.test_metric._get_name()}', test_output, on_epoch=True, on_step=False,
                     batch_size=self.batch_size)
        # update the conf mat
        self.conf_mat_test.update(y_hat, batch.y)
        # update the softmax predictions
        pred = torch.cat((batch.file_name_ind.unsqueeze(dim=1), batch.y.unsqueeze(dim=1), pred), dim=1)
        self.softmax_output.update(pred)

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        if self.compute_conf_test:
            self._create_and_save_conf_mat(self.conf_mat_test.compute().cpu().numpy(), f'test', save_csv=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)

        lr_scheduler = self.hparams.lr_scheduler
        if lr_scheduler == 'StepLR':
            if self.hparams.step_size is not None and self.hparams.step_size > 0:
                step_size = self.hparams.step_size
            else:
                step_size = self.hparams.epochs // 5
                if step_size == 0:
                    step_size = self.hparams.epochs
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
        elif 'CyclicLR' == lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                          mode='triangular2',
                                                          base_lr=self.hparams.learning_rate * 0.0001,
                                                          max_lr=self.hparams.learning_rate,
                                                          step_size_up=self.hparams.epochs // 4, cycle_momentum=False)
        elif lr_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        elif lr_scheduler == None:
            scheduler = None
        else:
            scheduler = eval(f'torch.optim.lr_scheduler.{lr_scheduler}')()

        return [optimizer], [scheduler]

    def _create_and_save_conf_mat(self, conf_mat, postfix: str, save_csv=False):
        run_id = '' if self.run_id is None else f'_run{self.run_id}'
        cv_id = '' if self.cv_ind is None else f'_cv{self.cv_ind}'
        conf_mat_name = f'conf_mat_{postfix}{run_id}{cv_id}'
        # create pandas dataframe
        df_cm = pd.DataFrame(conf_mat)

        # if of a feasible size, make an image and save it
        if df_cm.shape[0] <= 50:
            # empty image
            plt.figure(figsize=(20, 20))
            # create the cm
            sn.heatmap(df_cm, annot=True, fmt=".0f")
            # save plot as normal image on the disc
            plt.savefig(fname=self.output_path_analysis / (conf_mat_name + '.png'), format='png')
            # getting the plot as pil image according to
            # https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image/8598881
            for logger in self.loggers:
                if logger.__class__ is pl.loggers.WandbLogger:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    img = Image.open(buf)
                    logger.log_image(key='conf_mat', images=[img], caption=[f'{conf_mat_name}.'])
                    buf.close()
                    plt.close()

                wandb.log({conf_mat_name: wandb.Table(data=df_cm)})

            if save_csv:
                df_cm.to_csv(self.output_path_analysis / (conf_mat_name + '.csv'))

        else:
            df_cm.to_csv(self.output_path_analysis / (conf_mat_name + '.csv'))
            for logger in self.loggers:
                if logger.__class__ is pl.loggers.WandbLogger:
                    wandb.log({conf_mat_name: wandb.Table(data=df_cm)})
