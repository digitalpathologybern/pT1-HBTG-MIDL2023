# Utils
import argparse

# Torch
import torch


# base class
class BaseCLArguments:
    def __init__(self):
        # Create parser
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              description='Template for training a network on a dataset')
        # Add all options
        self._general_parameters()
        self._data_options()
        self._training_options()
        self._model_options()
        self._optimizer_options()
        self._system_options()
        self._wandb_options()

    def get_parser(self):
        """ Parse the command line arguments provided

        Parameters
        ----------
        args : str
            None, if set its a string which encloses the CLI arguments
            e.g. "--runner-class image_classification --output-folder log --dataset-folder datasets/MNIST"

        Returns
        -------
        args : dict
            Dictionary with the parsed arguments
        parser : ArgumentParser
            Parser used to process the arguments
        """
        return self.parser

    def _general_parameters(self):
        """ General options """
        parser_general = self.parser.add_argument_group('GENERAL', 'General Options')
        parser_general.add_argument('--experiment-name',
                                    type=str,
                                    default=None,
                                    help='provide a meaningful and descriptive name to this run')
        parser_general.add_argument('--input-folder',
                                    type=str,
                                    help='location of the dataset on the machine e.g root/data',
                                    required=False)
        parser_general.add_argument('--output-folder',
                                    type=str,
                                    default='./output/',
                                    help='where to save all output files.', )
        parser_general.add_argument('--multi-run',
                                    type=int,
                                    default=None,
                                    help='run main N times with different random seeds')
        parser_general.add_argument('--ignoregit',
                                    action='store_true',
                                    help='Run irrespective of git status.')
        parser_general.add_argument('--seed',
                                    type=int,
                                    default=None,
                                    help='random seed')
        parser_general.add_argument('--test-only',
                                    default=False,
                                    action='store_true',
                                    help='Skips the training phase.')
        parser_general.add_argument('--config-json',
                                    default=None,
                                    type=str,
                                    help='Path to config json instead of/additionally to using CLI arguments. '
                                         'Parameters in json override CLI arguments, if specified in both.')

    def _system_options(self):
        """ System options """
        parser_system = self.parser.add_argument_group('SYS', 'System Options')
        parser_system.add_argument('--gpu-id',
                                   type=int, nargs='*',
                                   default=None,
                                   help='which GPUs to use for training (use all by default)')
        parser_system.add_argument('--no-cuda',
                                   action='store_true',
                                   default=False,
                                   help='run on CPU')
        parser_system.add_argument('-j', '--workers',
                                   type=int,
                                   default=4,
                                   help='workers used for train/val loaders')

    def _data_options(self):
        """ Defines all parameters relative to the data. """
        parser_data = self.parser.add_argument_group('DATA', 'Dataset Options')
        parser_data.add_argument('--disable-databalancing-loss',
                                 default=False,
                                 action='store_true',
                                 help='Suppress data balancing in loss function')
        parser_data.add_argument('--rebuild-dataset', '--rebuild',
                                 default=False,
                                 action='store_true',
                                 help='Set to False if you want to load the dataset from /processed')
        parser_data.add_argument('--balanced-loading-train-val',
                                 default=False,
                                 action='store_true',
                                 help='Set if smaller classes in train and validation set should be upsampled.')

    def _training_options(self):
        """ Training options """
        # List of possible custom models already implemented
        parser_train = self.parser.add_argument_group('TRAIN', 'Training Options')
        parser_train.add_argument('--model-name', '--model',
                                  dest='model_name',
                                  type=str,
                                  help='which model to use for training')
        parser_train.add_argument('-b', '--batch-size',
                                  dest='batch_size',
                                  type=int,
                                  default=128,
                                  help='input batch size for training')
        parser_train.add_argument('--epochs',
                                  type=int,
                                  default=5,
                                  help='how many epochs to train')
        # parser_train.add_argument('--load-model',
        #                           type=str,
        #                           default=None,
        #                           help='path to latest checkpoint or'
        #                                'use pre-trained models from the modelzoo')
        # parser_train.add_argument('--validation-interval',
        #                           type=int,
        #                           default=1,
        #                           help='run evaluation on validation set every N epochs')
        parser_train.add_argument('--cross-val',
                                  default=None,
                                  type=int,
                                  help='specify how many splits there are for cross validation, if they are organized'
                                       'in folders')
        parser_train.add_argument('--split-json',
                                  default=None,
                                  type=str,
                                  help='specify path to json file that contains the dataset split. Expected json '
                                       'structure see ReadMe. Overrides')


    def _optimizer_options(self):
        """ Options specific for optimizers """
        # List of possible optimizers already implemented in PyTorch
        optimizer_options = [name for name in torch.optim.__dict__ if callable(torch.optim.__dict__[name])]
        lrscheduler_options = [name for name in torch.optim.lr_scheduler.__dict__ if
                               callable(torch.optim.lr_scheduler.__dict__[name])]
        parser_optimizer = self.parser.add_argument_group('OPTIMIZER', 'Optimizer Options')

        parser_optimizer.add_argument('--lr_scheduler',
                                      choices=lrscheduler_options,
                                      default='StepLR',
                                      help='optimizer to be used for training')
        parser_optimizer.add_argument('--momentum',
                                      type=float,
                                      default=0.9,
                                      help='momentum (parameter for the optimizer)')
        # parser_optimizer.add_argument('--dampening',
        #                               type=float,
        #                               default=0,
        #                               help='dampening (parameter for the SGD)')
        parser_optimizer.add_argument('--wd', '--weight-decay',
                                      type=float,
                                      dest='weight_decay',
                                      default=0,
                                      help='weight_decay coefficient, also known as L2 regularization')
        parser_optimizer.add_argument('--smoothing',
                                      type=float,
                                      default=0.2,
                                      help='label smoothing used in the loss function')
        parser_optimizer.add_argument('--lr', '--learning-rate',
                                      type=float,
                                      default=0.001,
                                      dest='learning_rate',
                                      help='learning rate to be used for training')
        parser_optimizer.add_argument('--dropout',
                                      type=float,
                                      default=0.3,
                                      dest='dropout',
                                      help='dropout rate to use')
        parser_optimizer.add_argument('--step-size',
                                      type=int,
                                      default=None,
                                      help='decays the learning rate by 0.01 every step_size epochs')

    def _model_options(self):
        """ Options for the model """
        parser_model = self.parser.add_argument_group('MODEL', 'Model Options')
        parser_model.add_argument('--ablate',
                                  action='store_true',
                                  default=False,
                                  help='if set, removes the last layers (usually the classifiction layer) of the model')
        parser_model.add_argument('--nb-neurons',
                                  type=int,
                                  default=128,
                                  help='Number of hidden representations per graph convolution layer')
        # parser_model.add_argument('--add-neurons-mlp',
        #                           type=int,
        #                           default=0,
        #                           help='Number of additional hidden representations per mlp layer')
        parser_model.add_argument('--num-layers',
                                  type=int,
                                  default=3,
                                  help='Number of graph convolution layers (does not work for every model!)')
        parser_model.add_argument('--normalization', '--norm',
                                  type=str, metavar='norm',
                                  default=None,
                                  help='Normalization layer to use in the GNN')
        parser_model.add_argument('--weight-init', '--weight-initializer',
                                  type=str, metavar='weight_initializer',
                                  default=None,
                                  help='Weight initialisation to use')

    # def _criterion_options(self):
    #     """ Options specific for optimizers """
    #     parser_optimizer = self.parser.add_argument_group('CRITERION', 'Criterion Options')
    #     parser_optimizer.add_argument('--criterion-name',
    #                                   default='CrossEntropyLoss',
    #                                   help='criterion to be used for training')

    def _wandb_options(self):
        """ WandB options"""

        parser_wandb = self.parser.add_argument_group('WANDB', 'WandB Options')
        parser_wandb.add_argument('--wandb-project',
                                  type=str,
                                  default=None,
                                  required=False,
                                  help='name of your wandb project')
        parser_wandb.add_argument('--wandb-sweep',
                                  default=False,
                                  action='store_true',
                                  help='Use this to enable wandb sweeps')


class GraphClassificationGxlCLArguments(BaseCLArguments):
    def __init__(self):
        # Add additional options
        super().__init__()

        self._gxl_graph_neural_network_options()

    def _gxl_graph_neural_network_options(self):
        """ Options used to run graph neural networks"""
        parser_graph_neural_network = self.parser.add_argument_group('GXL_Dataset',
                                                                     'Graph Neural Network Options for a GXL dataset')
        # parser_graph_neural_network.add_argument('--add-coordinates',
        #                                          default=True,
        #                                          action='store_false',
        #                                          help='Set if node positions should be used as a feature (assumes the features are present'
        #                                               'as "x" and "y" in the gxl file)')
        parser_graph_neural_network.add_argument('--features-to-use',
                                                 type=str,
                                                 help='Specify features that should be used like "NodeFeatureName1,NodeFeatureName2,EdgefeatureName1"')
        parser_graph_neural_network.add_argument('--categorical-features-json',
                                                 type=str,
                                                 help="Path to json file: dictionary with first level 'edge' and/or 'node' "
                                                      "and then a list of the attribute names that need to be one-hot encoded ( = are categorical) e.g. {'node': ['symbol'], 'edge': ['valence']}}")
        parser_graph_neural_network.add_argument('--cxl',
                                                 default=False,
                                                 action='store_true',
                                                 help='Set if there are cxl files that specify the dataset split (train.cxl, val.cxl, test.cxl)')
        parser_graph_neural_network.add_argument('--add-mlp-feature-csv',
                                                 type=str,
                                                 default=None,
                                                 help='Path to csv file that contains additional features to be considered for the classification MLP')
