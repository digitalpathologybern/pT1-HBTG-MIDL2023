from torch_geometric.nn.conv import (GATConv, GATv2Conv, GINConv, MessagePassing)
from torch_geometric.nn.models import MLP

from model_modules.graph_classifiers.BasicClassificationGNN import \
    BasicClassificationGNN  # overwrites the one from pytorch geometric
from model_modules.graph_classifiers.SAGEConvMod import SAGEConv

from model_modules.registry import Model


class GraphSAGE(BasicClassificationGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.
    See: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GraphSAGE
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, weight_initializer=self.weight_init)


class GIN(BasicClassificationGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.
    See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :obj:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        mlp = MLP(in_channels=in_channels, hidden_channels=out_channels,
                  out_channels=out_channels, num_layers=2, act='leaky_relu', dropout=self.dropout)
        # mlp = torch.nn.Sequential(Linear(self.in_channels, self.out_channels, weight_initializer=self.weight_init),
        #                           torch.nn.LeakyReLU(),
        #                           Linear(self.in_channels, self.out_channels, weight_initializer=self.weight_init))
        return GINConv(mlp)


class GAT(BasicClassificationGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
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
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def __init__(self, edge_dim: int = None, **kwargs):
        self.edge_dim = edge_dim
        super().__init__(**kwargs)

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and self.out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat, edge_dim=self.edge_dim,
                    dropout=self.dropout)


# GraphSAGE -----------------------------------------------------------------------------------------------------------

@Model
def graphsage(num_features, output_channels, num_layers=3, nb_neurons=128, **kwargs):
    model = GraphSAGE(in_channels=num_features, hidden_channels=nb_neurons, num_layers=num_layers,
                      out_channels=output_channels, **kwargs)
    model.hparam = {'message_passing_fct_name': 'graphsage',
                    'num_features': num_features,
                    'output_channels': output_channels,
                    'num_layers': num_layers,
                    'nb_neurons': nb_neurons,
                    'model_name': model.__repr__()}
    return model


@Model
def graphsage_jk(num_features, output_channels, num_layers=3, nb_neurons=128, **kwargs):
    model = GraphSAGE(in_channels=num_features, hidden_channels=nb_neurons, num_layers=num_layers,
                      out_channels=output_channels, jk='cat', **kwargs)
    model.hparam = {'message_passing_fct_name': 'graphsage_jk',
                    'num_features': num_features,
                    'output_channels': output_channels,
                    'num_layers': num_layers,
                    'nb_neurons': nb_neurons,
                    'jk': 'cat',
                    'model_name': model.__repr__()}
    return model


# GIN -----------------------------------------------------------------------------------------------------------------

@Model
def gin(num_features, output_channels, num_layers=3, nb_neurons=128, **kwargs):
    model = GIN(in_channels=num_features, hidden_channels=nb_neurons, num_layers=num_layers,
                out_channels=output_channels, **kwargs)
    model.hparam = {'message_passing_fct_name': 'gin',
                    'num_features': num_features,
                    'output_channels': output_channels,
                    'num_layers': num_layers,
                    'nb_neurons': nb_neurons,
                    'model_name': model.__repr__()}
    return model


@Model
def gin_jk(num_features, output_channels, num_layers=3, nb_neurons=128, **kwargs):
    model = GIN(in_channels=num_features, hidden_channels=nb_neurons, num_layers=num_layers,
                out_channels=output_channels, jk='cat', **kwargs)
    model.hparam = {'message_passing_fct_name': 'gin_jk',
                    'num_features': num_features,
                    'output_channels': output_channels,
                    'num_layers': num_layers,
                    'nb_neurons': nb_neurons,
                    'jk': 'cat',
                    'model_name': model.__repr__()}
    return model


# GraphSAGE -----------------------------------------------------------------------------------------------------------

@Model
def graphsage(num_features, output_channels, num_layers=3, nb_neurons=128, **kwargs):
    model = GraphSAGE(in_channels=num_features, hidden_channels=nb_neurons, num_layers=num_layers,
                      out_channels=output_channels, **kwargs)
    model.hparam = {'message_passing_fct_name': 'graphsage',
                    'num_features': num_features,
                    'output_channels': output_channels,
                    'num_layers': num_layers,
                    'nb_neurons': nb_neurons,
                    'model_name': model.__repr__()}
    return model


@Model
def graphsage_jk(num_features, output_channels, num_layers=3, nb_neurons=128, **kwargs):
    model = GraphSAGE(in_channels=num_features, hidden_channels=nb_neurons, num_layers=num_layers,
                      out_channels=output_channels, jk='cat', **kwargs)
    model.hparam = {'message_passing_fct_name': 'graphsage_jk',
                    'num_features': num_features,
                    'output_channels': output_channels,
                    'num_layers': num_layers,
                    'nb_neurons': nb_neurons,
                    'jk': 'cat',
                    'model_name': model.__repr__()}
    return model


# GAT -----------------------------------------------------------------------------------------------------------

@Model
def gatv2(num_features, output_channels, num_layers=3, nb_neurons=128, edge_dim=None, **kwargs):
    model = GAT(in_channels=num_features, hidden_channels=nb_neurons, v2=True, num_layers=num_layers,
                out_channels=output_channels, edge_dim=edge_dim, **kwargs)
    model.hparam = {'message_passing_fct_name': 'GAT',
                    'num_features': num_features,
                    'edge_dim': edge_dim,
                    'output_channels': output_channels,
                    'num_layers': num_layers,
                    'nb_neurons': nb_neurons,
                    'model_name': model.__repr__()}
    return model


@Model
def gatv2_jk(num_features, output_channels, num_layers=3, nb_neurons=128, edge_dim=None, **kwargs):
    model = GAT(in_channels=num_features, hidden_channels=nb_neurons, num_layers=num_layers,
                out_channels=output_channels, jk='cat', v2=True, **kwargs)
    model.hparam = {'message_passing_fct_name': 'GAT_jk',
                    'num_features': num_features,
                    'edge_dim': edge_dim,
                    'output_channels': output_channels,
                    'num_layers': num_layers,
                    'nb_neurons': nb_neurons,
                    'jk': 'cat',
                    'model_name': model.__repr__()}
    return model
