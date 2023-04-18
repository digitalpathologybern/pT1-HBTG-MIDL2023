import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.nn.glob import global_add_pool
from torch_geometric.nn.dense.linear import Linear

from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge

from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)

# adapted from base class from
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html


class BasicClassificationGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        jk: Optional[str] = None,
        act: Union[str, Callable, None] = "LeakyReLU",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        add_neurons_mlp: int = 0,
        weight_init: str = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.add_feat = add_neurons_mlp > 0

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.weight_init = weight_init

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        self.convs.append(
            self.init_conv(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))

        self.norms = None
        self.norm = norm
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                hidden_channels,
                **(norm_kwargs or {}),
            )
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm_layer))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                hidden_channels = num_layers * hidden_channels
            else:
                hidden_channels = hidden_channels

        # classification layer
        self.lin1 = Linear(hidden_channels + add_neurons_mlp, hidden_channels*2, weight_initializer=weight_init)
        self.lin1 = Linear(hidden_channels + add_neurons_mlp, hidden_channels*2, weight_initializer=weight_init)
        self.lin2 = Linear(hidden_channels*2, hidden_channels, weight_initializer=weight_init)
        self.lin_class = Linear(hidden_channels, out_channels, weight_initializer=weight_init)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        self.lin.reset_parameters()
        self.lin_class.reset_parameters()

    def forward(self, data, batch_size, *args, **kwargs) -> Tensor:
        """"""
        # prepare the data
        x, edge_index = data.x, data.edge_index

        if self.add_feat:
            add_feat = data.add_mlp_features
        else:
            add_feat = None
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x

        # read-out
        if hasattr(data, 'node_mask'):  # in case of node dropout
            x_graph = global_add_pool(x[data.node_mask], data.batch[data.node_mask], size=batch_size)
        else:
            x_graph = global_add_pool(x, data.batch, size=batch_size)

        # add the additional features to be considered by the MLP, if available
        if add_feat is not None:
            # last batch can be of unequal size, zero pad
            if x_graph.size()[0] != add_feat.size()[0]:
                add_feat = F.pad(add_feat, pad=(0, 0, 0, batch_size-add_feat.size()[0]))
            x_graph = torch.cat((x_graph, add_feat), 1)

        x_graph = self.act(self.lin1(x_graph))
        x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)
        x_graph = self.act(self.lin2(x_graph))
        x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)
        x_output = self.lin_class(x_graph)
        return x_output

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers}), jk={self.jk_mode}'
                f'dropout={self.dropout}, act_fct={self.act}, norm={self.norm},'
                f'weight_initializer={self.weight_init}')

