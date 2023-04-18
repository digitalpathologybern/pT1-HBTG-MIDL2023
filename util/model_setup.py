import os
import torch
import torch.backends.cudnn as cudnn
import logging

import model_modules


def setup_model(model_name: str, no_cuda: bool, num_outputs: int, num_features: int, gpu_id: list,
                load_model: bool = None, strict: bool = True, add_neurons_mlp: int = 0,
                num_edge_features: int = None, **kwargs):
    """Setup the model, load and move to GPU if necessary

    Parameters
    ----------
    model_name : string
        Name of the model
    no_cuda : bool
        Specify whether to use the GPU or not
    num_outputs : int
        How many different classes there are in our problem. Used for loading the model.
    load_model : string
        Path to a saved model
    strict : bool
        Enforces key match between loaded state_dict and model definition

    Returns
    -------
    model : DataParallel
        The model
    """
    # Initialize the model
    logging.info('Setting up model {}'.format(model_name))

    # has to be converted, because for the colors-3 dataset it gets loaded as a float
    num_features = int(num_features)
    num_outputs = int(num_outputs)
    if model_name not in model_modules.__dict__.keys():
        logging.error(f'Model name "{model_name}" not recognized.')
    model = model_modules.__dict__[model_name](output_channels=num_outputs, num_features=num_features,
                                               add_neurons_mlp=add_neurons_mlp, edge_dim=num_edge_features,
                                               **kwargs)

    # Transfer model to GPU
    if not no_cuda:
        logging.info('Transfer model to GPU')
        model = model.to(torch.device(f'cuda:{gpu_id[0]}'))
        cudnn.benchmark = True

    return model

