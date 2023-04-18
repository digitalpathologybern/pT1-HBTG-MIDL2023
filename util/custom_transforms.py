import logging
import sys
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform, LinearTransformation

try:
    import torch_cluster  # noqa

    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.subgraph import subgraph

####################################################################################################
####################################################################################################
# Transforms that change the graph structure
####################################################################################################
####################################################################################################


class CentralNode(BaseTransform):
    """
    Adds a fully connected node to the graph (i.e. a node that is connected to all other nodes in the graph). The
    features are initialized with the mean of all node features in the graph.
    """

    def __call__(self, data):
        # add edges to all other nodes
        additional_edges = torch.tensor([[data.num_nodes] * data.num_nodes, list(range(data.num_nodes))])
        data.edge_index = torch.cat((data.edge_index, additional_edges), 1)

        # add the central node
        central_node_features = torch.mean(data.x, 0)
        data.x = torch.cat((data.x, central_node_features.unsqueeze(0)), 0)

        # give it a position
        if data.pos is not None:
            central_node_pos = torch.mean(data.pos, 0)
            data.pos = torch.cat((data.pos, central_node_pos.unsqueeze(0)), 0)

        data.num_nodes = data.x.size(0)
        return data


class ComplGraph(BaseTransform):
    """
    Adds the complementary graph to the data object
    """

    def __call__(self, data):
        data.keys.append('dual_edge_index')

        edge_ind = np.array(data.edge_index)
        node_ind_list = list(range(data.num_nodes))
        from_to_index_dict = {i: [] for i in node_ind_list}
        _ = {from_to_index_dict[edge_ind[0][i]].append(edge_ind[1][i]) for i in range(len(edge_ind[0]))}

        from_to_dual_graph_dict = {i: [] for i in node_ind_list}
        for from_indx, to_list in from_to_index_dict.items():
            from_to_dual_graph_dict[from_indx] = [x for x in node_ind_list if x not in to_list]

        from_ = []
        to_ = []
        for from_ind, to_list in from_to_dual_graph_dict.items():
            from_ += [from_ind] * len(to_list)
            to_ += to_list

        data.dual_graph_edge_index = torch.tensor([from_, to_], dtype=torch.int64)

        return data


class NodeDrop(BaseTransform):
    # adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/dropout.html#dropout_node
    def __init__(self, p: float = 0.5, training: bool = True):
        if p < 0. or p > 1.:
            raise ValueError(f'Dropout probability has to be between 0 and 1 (got {p}')
        else:
            self.p = p
        self.training = training

    def __call__(self, data):
        edge_index = data.edge_index
        num_nodes = maybe_num_nodes(edge_index, data.num_nodes)

        if not self.training or self.p == 0.0:
            node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
            edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
            return edge_index, edge_mask, node_mask

        prob = torch.rand(num_nodes, device=edge_index.device)
        node_mask = prob > self.p
        edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                            num_nodes=num_nodes,
                                            return_edge_mask=True)
        data.edge_index = edge_index
        data.num_edges = edge_index.size(1)
        data.node_mask = node_mask
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RemoveIsolatedNodes(BaseTransform):
    r"""Removes isolated nodes from the graph
    (functional name: :obj:`remove_isolated_nodes`)."""
    # adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/remove_isolated_nodes.html#RemoveIsolatedNodes
    def __call__(self, data):
        # if we have no edges, we keep all the nodes as they are
        if data.edge_index.size()[1]==0:
            return data

        # Gather all nodes that occur in at least one edge (across all types):
        n_id_dict = defaultdict(list)
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            if store._key is None:
                src = dst = None
            else:
                src, _, dst = store._key

            n_id_dict[src].append(store.edge_index[0])
            n_id_dict[dst].append(store.edge_index[1])

        n_id_dict = {k: torch.cat(v).unique() for k, v in n_id_dict.items()}

        n_map_dict = {}
        for store in data.node_stores:
            if store._key not in n_id_dict:
                n_id_dict[store._key] = torch.empty((0, ), dtype=torch.long)

            idx = n_id_dict[store._key]
            mapping = idx.new_zeros(data.num_nodes)
            mapping[idx] = torch.arange(idx.numel(), device=mapping.device)
            n_map_dict[store._key] = mapping

        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            if store._key is None:
                src = dst = None
            else:
                src, _, dst = store._key

            row = n_map_dict[src][store.edge_index[0]]
            col = n_map_dict[dst][store.edge_index[1]]
            store.edge_index = torch.stack([row, col], dim=0)

        for store in data.node_stores:
            for key, value in store.items():
                if key == 'num_nodes':
                    store.num_nodes = n_id_dict[store._key].numel()

                elif store.is_node_attr(key):
                    store[key] = value[n_id_dict[store._key]]

        return data


####################################################################################################
####################################################################################################
# Adding and removing node features
####################################################################################################
####################################################################################################

class AddPosToNodeFeature(BaseTransform):
    r"""
    Adds the position of the node as a node feature
    """

    def __call__(self, data):
        data.x = torch.cat((data.pos, data.x), 1)
        data.num_node_features += 2
        return data


####################################################################################################
####################################################################################################
# Normalizations
####################################################################################################
####################################################################################################

class CenterPos(BaseTransform):
    r"""
    Centers x and y coordinate position around the origin.
    """

    def __call__(self, data):
        data.pos = data.pos - data.pos.mean(dim=-2, keepdim=True)
        return data


class ZNormalisationOverDataset(BaseTransform):
    """
    z-normalises the node attributes (continuous values, the categorical node features are not normalized)
    based on the standard deviation and mean of the whole training dataset.

    z-score = (x-mean)/std

    If there are features (categorical, one hot encoded) and attributes (continuous), the attributes are always
    before the the features

    Args:
        std_mean_dict: dict
            dictionary that contains the mean and standard deviation for all the node and edge
            features (calculated on the training set).
        num_node_attributes: int
        num_edge_attributes: int
    """

    def __init__(self, mean_std_dict: dict, num_node_attributes: int = 0, num_edge_attributes: int = 0):
        self.nodef_mean = mean_std_dict['node_features']['mean']
        self.nodef_std = mean_std_dict['node_features']['std']
        self.edgef_mean = mean_std_dict['edge_features']['mean']
        self.edgef_std = mean_std_dict['edge_features']['std']

        self.num_node_attributes = num_node_attributes
        self.num_edge_attributes = num_edge_attributes

    def __call__(self, data):
        # ensure that the mean and std have the same length as the feature matrix (if not, dataset should be rebuilt)
        if len(self.nodef_mean) != data.num_node_features or len(self.edgef_mean) != data.num_edge_features:
            logging.error(
                "Computed feature mean/std does not match feature matrix size! You need to rebuild your dataset!")
            sys.exit(-1)

        # check if we have node attributes, if yes z-normalize
        if self.num_node_attributes > 0:
            for i in range(self.num_node_attributes):
                data.x[:, i] = self.z_normalize(data.x[:, i], mean=self.nodef_mean[i], std=self.nodef_std[i])
        # check if we have edge attributes, if yes z-normalize
        if self.num_edge_attributes > 0:
            for i in range(self.num_edge_attributes):
                data.x[:, i] = self.z_normalize(data.x[:, i], mean=self.edgef_mean[i], std=self.edgef_std[i])

        return data

    @staticmethod
    def z_normalize(features, mean, std):
        # features can be a single value, list, numpy array or torch array
        return (features - mean) / std


class ZNormalisationPerGraph(BaseTransform):
    """
    z-normalises the node attributes (continuous values, the categorical node features are not normalized).

    z-score = (x-mean)/std

    If there are features (categorical, one hot encoded) and attributes (continous), the attributes are always
    before the the features

    Args:
        num_node_attributes: int
        num_edge_attributes: int

    """

    def __init__(self, num_node_attributes: int = 0, num_edge_attributes: int = 0):
        self.num_node_attributes = num_node_attributes
        self.num_edge_attributes = num_edge_attributes

    def __call__(self, data):
        # check if we have node attributes, if yes z-normalize
        if self.num_node_attributes > 0:
            # computing the mean and std of each feature

            for i in range(self.num_node_attributes):
                data.x[:, i] = self.z_normalize(data.x[:, i])
        # check if we have edge attributes, if yes z-normalize
        if self.num_edge_attributes > 0:
            for i in range(self.num_edge_attributes):
                data.x[:, i] = self.z_normalize(data.x[:, i])

        return data

    @staticmethod
    def z_normalize(features):
        # features can be a single value, list, numpy array or torch array
        return (features - np.mean(features)) / np.std(features)
