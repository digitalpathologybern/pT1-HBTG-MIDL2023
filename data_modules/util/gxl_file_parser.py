import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import sys
import pandas as pd

from data_modules.util.custom_exception import InvalidFileException
import logging


class ParsedGxlGraph:
    def __init__(self, path_to_gxl: str, class_label: int, subset: str = None, remove_coordinates: bool = True,
                 center_coordinates: bool = False, class_name=None):
        """
        This class contains all the information encoded in a single gxl file = one graph
        Parameters
        ----------
        subset : str
            either 'test', 'val' or 'train'
        class_label : int
            class label of the graph
        class_name: str
            the class label as a string (useful when the int is an encoding for a name)
        path_to_gxl: str
            path to the gxl file
        """
        self.filepath = path_to_gxl
        self.subset = subset
        self.class_label = class_label
        self.class_name = class_name
        self.remove_coordinates = remove_coordinates
        self.center_coordinates = center_coordinates

        self.filename = os.path.basename(self.filepath)
        # name of the gxl file (without the ending)
        self.file_id = self.filename[:-4]

        # parsing the gxl
        # sets up the following properties: node_features, node_feature_names, edges, edge_features, edge_feature_names,
        # node_position, graph_id, edge_ids_present and edgemode
        self.setup_graph_features()

    @property
    def class_name(self):
        return self._class_name

    @class_name.setter
    def class_name(self, class_name):
        if class_name is not None:
            self._class_name = class_name
        else:
            self._class_name = self. class_label

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, path_to_gxl):
        if not os.path.isfile(path_to_gxl):
            logging.error(f'File {path_to_gxl} does not exist.')
            sys.exit(-1)
        self._filepath = path_to_gxl

    @property
    def subset(self) -> str:
        return self._subset

    @subset.setter
    def subset(self, subset):
        # if set ensure that it is a valid arguments
        # if subset and subset not in ['train', 'val', 'test']:
        #     logging.error(f"Subset has to be specified as either 'train', 'val' or 'test'")
        #     sys.exit(-1)
        self._subset = subset

    @property
    def nb_of_nodes(self):
        return len(self.node_features)

    @property
    def nb_of_edges(self):
        return len(self.edge_features)

    def one_hot_encode_node_features(self, feature_encoding):
        """
        Update the nodes with the encoding for the categorical features
        feature_encoding: list
            contains the numerical encoding for each feature
        """
        one_hot_df = pd.DataFrame()
        df = pd.DataFrame(self.node_features, columns=self.node_feature_names)
        feature_names = []
        for feature_name, encoding in feature_encoding.items():
            # create the onehot encoding
            encoding_dict = {k: ','.join([str(i) for i in v]) for k, v in encoding.items()}
            one_hot = df[feature_name].replace(encoding_dict)
            one_hot = one_hot.str.split(',', 1, expand=True).astype('int')  # TODO: fix warning?
            one_hot_df = pd.concat([one_hot_df, one_hot])
            df.drop(feature_name, axis=1, inplace=True)
            self.node_feature_names.remove(feature_name)
            feature_names = feature_names + [f'{feature_name}_{k}' for k in encoding_dict.keys()]

        # update the node features
        self.node_features = pd.concat([df, one_hot_df], axis=1).values.tolist()
        self.node_feature_names = self.node_feature_names + feature_names

    def one_hot_encode_edge_features(self, feature_encoding):
        """
        Update the edges with the encoding for the categorical features
        feature_encoding: list
            contains the numerical encoding for each feature
        """
        feature_names = []
        one_hot_df = pd.DataFrame()
        df = pd.DataFrame(self.edge_features, columns=self.edge_feature_names)
        for feature_name, encoding in feature_encoding.items():
            # create the onehot encoding
            encoding_dict = {k: ','.join([str(i) for i in v]) for k, v in encoding.to_dict('list').items()}
            one_hot = df[feature_name].replace(encoding_dict)
            one_hot = one_hot.str.split(',', 1, expand=True).astype('int')
            one_hot_df = pd.concat([one_hot_df, one_hot])
            df.drop(feature_name, axis=1, inplace=True)
            self.edge_feature_names.remove(feature_name)
            feature_names = feature_names + [f'{feature_name}_{k}' for k in encoding_dict.keys()]

        # update the node features
        self.edge_features = pd.concat([df, one_hot_df], axis=1).values.tolist()
        self.edge_feature_names = self.edge_feature_names + feature_names

    def setup_graph_features(self):
        """
        Parses the gxl file and sets the following graph properties
        - graph info: graph_id, edge_ids_present and edgemode
        - node: node_features, node_feature_names, and node_position
        - edge: edges, edge_features and edge_feature_names

        """
        tree = ET.parse(self.filepath)
        root = tree.getroot()

        # verify that the file contains the expected attributes (node, edge, id, edgeids and edgemode)
        self.sanity_check(root)

        self.edges = self.get_edges(root)  # [[int, int]]
        self.node_feature_names, self.node_features = self.get_features(root, 'node')  # ([str], list)

        x_ind = self.node_feature_names.index('x')
        y_ind = self.node_feature_names.index('y')
        # Add the coordinates to their own variable
        if self.node_feature_names is not None and 'x' in self.node_feature_names and 'y' in self.node_feature_names:
            # if true, coordinates need to be centered (just in feature vector, xy-vector - xy(graph average)-vector)
            if self.center_coordinates and not self.remove_coordinates:
                x_mean = np.mean([node[x_ind] for node in self.node_features])
                y_mean = np.mean([node[y_ind] for node in self.node_features])
                for node in self.node_features:
                    node[x_ind] = node[x_ind] - x_mean
                    node[y_ind] = node[y_ind] - y_mean
            self.node_positions = [[node[x_ind], node[y_ind]] for node in self.node_features]
        else:
            self.node_positions = []  # [float / int]

        if self.remove_coordinates:
            self.node_feature_names.remove('x')
            self.node_feature_names.remove('y')
            for i in self.node_features:
                del i[x_ind:y_ind+1]

        self.edge_feature_names, self.edge_features = self.get_features(root, 'edge')  # ([str], list)
        self.graph_id, self.edge_ids_present, self.edgemode = self.get_graph_attr(root)  # (str, bool, str)

    def get_node_feature_values(self, feature) -> list:
        feature_ind = self.node_feature_names.index(feature)
        all_features = [nf[feature_ind] for nf in self.node_features]
        return all_features

    def get_edge_feature_values(self, feature) -> list:
        feature_ind = self.edge_features.index(feature)
        all_features = [[nf][feature_ind] for nf in self.edge_features]
        return all_features

    def get_features(self, root, mode):
        """
        get a list of the node features out of the element tree (gxl)

        Parameters
        ----------
        root: gxl element
        mode: str
            either 'edge' or 'node'

        Returns
        -------
        tuple ([str], [mixed values]])
            list of all node features for that tree
            ([feature name 1, feature name 2, ...],  [[feature 1 of node 1, feature 2 of node 1, ...], [feature 1 of node 2, ...], ...])
        """
        features_info = [[feature for feature in graph_element] for graph_element in root.iter(mode)]
        if len(features_info) > 0:
            feature_names = [i.attrib['name'] for i in features_info[0]]
        else:
            feature_names = []

        # check if we have features to generate
        if len(feature_names) > 0:
            features = [[self.decode_feature(value) for feature in graph_element for value in feature if feature.attrib['name'] in feature_names] for graph_element in root.iter(mode)]
        else:
            feature_names = None
            features = []

        return feature_names, features

    def sanity_check(self, root):
        """
        Check if files contain the expected content

        Parameters
        ----------
        root:

        Returns
        -------
        None
        """
        # check if node, edge, edgeid, edgemode, edgemode keyword exists
        if len([i.attrib for i in root.iter('graph')][0]) != 3:
            raise InvalidFileException

        if len([node for node in root.iter('node')]) == 0:
            logging.warning(f'File {os.path.basename(self.filepath)} is an empty graph!')
            raise InvalidFileException
        elif len([edge for edge in root.iter('edge')]) == 0:
            logging.warning(f'File {os.path.basename(self.filepath)} has no edges!')

    def normalize(self, mean_std):
        """
        This method normalizes the node and edge features (if present)

        Parameters§
        ----------
        graph: ParsedGxlGraph

        mean_std: dict
            dictionary containing the mean and standard deviation for the node and edge features
            {'node_features_':{'mean':X.X, 'std':X.X}, 'edge_features':{'mean':X.X, 'std':X.X}

        Returns
        ----------
        Normalized graph

        """
        def normalize(feature, mean, std):
            return (feature - mean) / std

        node_mean = mean_std['node_features']['mean']
        node_std = mean_std['node_features']['std']
        edge_mean = mean_std['edge_features']['mean']
        edge_std = mean_std['edge_features']['std']

        # check if graph is not empty
        if len(self.node_features) > 0:
            # normalize the node features
            for node_ind in range(len(self.node_features)):
                self.node_features[node_ind] = [
                    normalize(self.node_features[node_ind][i], node_mean[i], node_std[i])
                    for i in range(len(node_mean))]
            # normalize the edge features (except the coordinates
            for edge_ind in range(len(self.edge_features)):
                self.edge_features[edge_ind] = [
                    normalize(self.edge_features[edge_ind][i], edge_mean[i], edge_std[i])
                    for i in range(len(edge_mean))]

    @staticmethod
    def get_graph_attr(root) -> tuple:
        """
        Gets the information attributes of the whole graph:
        Parameters
        ----------
        root: gxl element
            root of ET tree

        Returns
        -------
        tuple (str, bool, str)
            ID of the graph, Edge IDs present (true / false), edge mode (directed / undirected)
        """
        graph = [i.attrib for i in root.iter('graph')]
        assert len(graph) == 1
        g = graph[0]
        return g['id'], g['edgeids'] == 'True', g['edgemode']

    @staticmethod
    def get_edges(root) -> list:
        """
        Get the start and end points of every edge and store them in a list of lists (from the element tree, gxl)
        Parameters
        ----------
        root: gxl element

        Returns
        -------
        [[int, int]]
            list of indices of connected nodes
        """
        edge_list = []

        regex = '_(\d+)$'
        start_points = [int(re.search(regex, edge.attrib['from']).group(1)) for edge in root.iter('edge')]
        end_points = [int(re.search(regex, edge.attrib['to']).group(1)) for edge in root.iter('edge')]
        assert len(start_points) == len(end_points)

        # move enumeration start to 0 if necessary
        if len(start_points) > 0 and len(end_points) > 0:
            if min(min(start_points, end_points)) > 0:
                shift = min(min(start_points, end_points))
                start_points = [x - shift for x in start_points]
                end_points = [x - shift for x in end_points]

            edge_list = [[start_points[i], end_points[i]] for i in range(len(start_points))]

        return edge_list

    @staticmethod
    def decode_feature(f) -> str:
        data_types = {'string': str,
                      'float': float,
                      'int': int}

        # convert the feature value to the correct data type as specified in the gxl
        return data_types[f.tag](f.text.strip())
