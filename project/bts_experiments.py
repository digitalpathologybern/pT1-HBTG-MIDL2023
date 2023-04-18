import logging

import torch_geometric
from torch_geometric.transforms import RandomShear, RandomScale, RandomRotate, RandomFlip

from util.arg_parsers import GraphClassificationGxlCLArguments
from util.model_setup import setup_model
from util.custom_transforms import CenterPos, AddPosToNodeFeature, NodeDrop
from data_modules.classification_gxl_dataset import GxlDataset
from project.experiment_template import Experiment
from runners.graph_classification import GraphClassification
from project import project_util
import pytorch_lightning as pl
from torchmetrics import F1Score, Precision, Recall, Specificity, AUROC, MetricCollection


class Bts(Experiment):
    def __init__(self, multi_run, **kwargs):
        super(Bts, self).__init__(multi_run=multi_run, **kwargs)
        # self.kwargs['weight_initializer'] = 'kaiming_uniform'
        # self.kwargs['norm'] = 'GraphNorm'
        self.set_metrics()

    # ------------
    # metrics
    # ------------
    def set_metrics(self):
        self.train_metric = MetricCollection({'F1Score_c': F1Score(num_classes=self.num_classes, average='none'),
                                              'Precision_c': Precision(num_classes=self.num_classes, average='none'),
                                              'Recall_c': Recall(num_classes=self.num_classes,
                                                                  average='none'),
                                              'F1Score': F1Score(average='macro', num_classes=self.num_classes),
                                              'Precision': Precision(average='macro', num_classes=self.num_classes),
                                              'Recall': Recall(average='macro', num_classes=self.num_classes),
                                              # 'Specificity': Specificity(average='macro',
                                              #                            num_classes=self.num_classes)
                                              })

        self.val_metric = MetricCollection({'F1Score_c': F1Score(num_classes=self.num_classes, average='none'),
                                            'Precision_c': Precision(num_classes=self.num_classes, average='none'),
                                            'Recall_c': Recall(num_classes=self.num_classes, average='none'),
                                            'F1Score': F1Score(average='macro', num_classes=self.num_classes),
                                            'Precision': Precision(average='macro', num_classes=self.num_classes),
                                            'Recall': Recall(average='macro', num_classes=self.num_classes),
                                            # 'Specificity': Specificity(average='macro',
                                            #                            num_classes=self.num_classes)
                                            })

        self.test_metric = MetricCollection({'F1Score_c': F1Score(num_classes=self.num_classes, average='none'),
                                             'Precision_c': Precision(num_classes=self.num_classes, average='none'),
                                             'Recall_c': Recall(num_classes=self.num_classes,
                                                                 average='none'),
                                             'F1Score': F1Score(average='macro', num_classes=self.num_classes),
                                             'Precision': Precision(average='macro', num_classes=self.num_classes),
                                             'Recall': Recall(average='macro', num_classes=self.num_classes),
                                             # 'Specificity': Specificity(average='macro',
                                             #                            num_classes=self.num_classes),
                                             # 'AUROC': AUROC(average='macro', num_classes=self.num_classes),
                                             # 'AUROC_c': AUROC(average=None, num_classes=self.num_classes)
                                             })

    # ------------
    # data
    # ------------
    def get_dataset_full(self):
        # dataset is not split into folders --> json file with splits
        return GxlDataset(pre_transform=self.get_pre_transforms(), split_json=self.split_json, **self.kwargs)

    def get_dataset_splits(self):
        # dataset is split in train / val / test folders
        train_ds = GxlDataset(subset='train', pre_transform=self.get_pre_transforms(), **self.kwargs)
        val_ds = GxlDataset(subset='val', pre_transform=self.get_pre_transforms(), **self.kwargs)
        test_ds = GxlDataset(subset='test', pre_transform=self.get_pre_transforms(),
                             **self.kwargs)
        return train_ds, val_ds, test_ds

    def get_train_transforms(self):
        return torch_geometric.transforms.Compose([
            RandomShear(shear=1.05),
            RandomScale(scales=(0.95, 1.05)),
            RandomRotate(degrees=360),
            CenterPos(),
            AddPosToNodeFeature(),
            NodeDrop(p=0.1)
        ])

    def get_val_transforms(self):
        return self.get_train_transforms()

    def get_test_transforms(self):
        return torch_geometric.transforms.Compose([CenterPos(),
                                                   AddPosToNodeFeature()])

    @staticmethod
    def get_pre_transforms():
        return None

    # ------------
    # model
    # ------------
    def get_model(self, run_id=None, cv_ind=None):
        # Initialize the model
        # check if we have additional features that are added after the read-out
        if self.train_ds.add_mlp_features is None:
            add_neurons = 0
        else:
            add_neurons = self.train_ds.add_mlp_features.columns.size

        logging.info('Setting up model {}'.format(self.kwargs['model_name']))
        # self.kwargs['norm'] = torch_geometric.nn.BatchNorm(self.kwargs['nb_neurons'])
        model = setup_model(num_outputs=self.num_classes, num_features=self.num_features, add_neurons_mlp=add_neurons,
                            num_edge_features=self.num_edge_features, **self.kwargs)
        model = GraphClassification(model=model, class_frequency=self.class_frequencies, num_classes=self.num_classes,
                                    run_id=run_id, transforms=self.get_train_transforms(),
                                    pre_transforms=self.get_pre_transforms(),
                                    train_metric=self.train_metric, val_metric=self.val_metric,
                                    test_metric=self.test_metric, cv_ind=cv_ind,
                                    **self.kwargs)

        return model

    def get_checkpoint_callback(self, run_id=None, cv_ind=None) -> pl.callbacks.ModelCheckpoint:
        # default is last model is saved
        run_tag = '' if run_id is None else f'_run{run_id}'
        cv_tag = '' if cv_ind is None else f'_cv{cv_ind}'
        checkpoint_filename = f'{self.kwargs["experiment_name"]}{run_tag}{cv_tag}'
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=self.kwargs['output_folder'],
                                                           filename=checkpoint_filename + '_{epoch:02d}_{val/loss:02f}',
                                                           monitor='val/loss', mode='min')
        return checkpoint_callback


if __name__ == '__main__':
    # ------------
    # args, setup and logging
    # ------------
    kwargs = project_util.get_kwargs(GraphClassificationGxlCLArguments())

    Bts(**kwargs).main()
