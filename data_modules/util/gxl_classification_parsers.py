import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import sys
import random
import logging
import pandas as pd

from data_modules.util.custom_exception import InvalidFileException
from data_modules.util.gxl_file_parser import ParsedGxlGraph


class ParsedGxlDataset:
    def __init__(self, path_to_dataset: str, categorical_features: dict, ignore_coordinates: bool = True):
        """
        This class creates a dataset object containing all the graphs parsed from gxl files as ParsedGxlGraph objects

        Parameters
        ----------
        ignore_coordinates : bool
            if true, coordinates are removed from the node features
        path_to_dataset: str
            path to the folder with gxl files
        subset: if specified, has to be either 'train', 'test' or 'val'
        """
        self.path_to_dataset = path_to_dataset
        self.name = os.path.basename(os.path.dirname(self.path_to_dataset))
        self.remove_coordinates = ignore_coordinates

        # get a list of the empty graphs
        self.invalid_files = []

        # parse the graphs
        self.graphs = self.get_graphs()
        self.class_names = list(set([g.class_label for g in self.graphs]))

        # get the node and edge features names at a higher level plus their data type
        # sets: node_feature_names, edge_feature_names, node_dtypes, and edge_dtypes
        self.categorical_features = categorical_features
        self.set_feature_names_and_types()

        # if there are string-based features, we need to encode them as a one-hot vector
        # If there are features (categorical, one hot encoded) and attributes (continuous), the attributes are always
        # before the the features.
        if self.node_feature_names and len(self.categorical_features['node']) > 0:
            self.one_hot_encode_nodes()
        if self.edge_feature_names and len(self.categorical_features['edge']) > 0:
            self.one_hot_encode_edges()

        self.set_feature_names_and_types()
        # get a list of all the filenames
        self.file_names = [g.filename for g in self.graphs]

    @property
    def file_ids(self):
        # get a list of all the filenames without the path and the extension
        return [g.file_id for g in self.graphs]

    @property
    def path_to_dataset(self) -> str:
        return self._path_to_dataset

    @path_to_dataset.setter
    def path_to_dataset(self, path_to_dataset):
        if not os.path.isdir(path_to_dataset):
            logging.error(f'Folder {path_to_dataset} does not exist.')
            sys.exit(-1)
        self._path_to_dataset = path_to_dataset

    @property
    def edge_mode(self) -> str:
        return self.graphs[0].edgemode

    @property
    def config(self) -> dict:
        # Setup the configuration dictionary
        config = {
            'dataset_name': self.name,
            'node_feature_names': self.node_feature_names,
            'edge_feature_names': self.edge_feature_names,
            'classes': self.class_names,
        }
        if hasattr(self, 'subset'):
            config['dataset_split'] = self.subset
        if hasattr(self, 'json_split'):
            config['json_split'] = self.json_split
        if hasattr(self, 'nodes_onehot'):
            config['one-hot_encoding_node_features'] = self.nodes_onehot
        if hasattr(self, 'edges_onehot'):
            config['one-hot_encoding_edge_features'] = self.edges_onehot
        if hasattr(self, 'categorical_features'):
            config['categorical_features'] = self.categorical_features
        if hasattr(self, 'class_encoding'):
            config['class_encoding'] = self.class_int_encoding
        return config

    def one_hot_encode_nodes(self):
        """
        This functions one-hot encodes the categorical node feature values and changes them in all the graphs
        (graph.node_features)

        sets self.nodes_onehot: one-hot encoding of the categorical node features {'feature name': {'feature value': encoding}
        """
        # get all the features
        all_features = {feature: [set(g.get_node_feature_values(feature)) for g in self.graphs] for feature in self.categorical_features['node']}
        for feature, all_f in all_features.items():
            all_features[feature] = sorted(list(set.union(*all_f)))
        # get the one-hot encodings
        encoding = {f: pd.get_dummies(l).to_dict('list') for f, l in all_features.items()}
        self.nodes_onehot = encoding
        # update the graphs
        for g in self.graphs:
            g.one_hot_encode_node_features(encoding)

    def one_hot_encode_edges(self):
        """
        This functions one-hot encodes the categorical edge feature values and changes them in all the graphs
        (graph.edge_features)

        sets self.edges_onehot: one-hot encoding of the categorical edge features {'feature name': {'feature value': encoding}
        """
        # get all the features
        all_features = {feature: [set(g.get_node_feature_values(feature)) for g in self.graphs] for feature in
                        self.categorical_features['node']}
        for feature, all_f in all_features.items():
            all_features[feature] = sorted(list(set.union(*all_f)))
        # get the one-hot encodings
        encoding = {f: pd.get_dummies(l) for f, l in all_features.items()}
        self.edges_onehot = encoding
        # update the graphs
        for g in self.graphs:
            g.one_hot_encode_edge_features(encoding)

    def set_feature_names_and_types(self):
        # get the node and edge feature names available at a higher level
        agraph = [g for g in self.graphs if len(g.node_features) > 0 and len(g.edges) > 0][0]
        self.node_feature_names = agraph.node_feature_names
        self.edge_feature_names = agraph.edge_feature_names

        # set node / edge feature data type and update self.categorical_features
        # if node type is a string, add them to self.categorical_features
        # nodes
        if len(agraph.node_features) > 0:
            self.node_dtypes = [type(dtype) for dtype in agraph.node_features[0]]
            assert len(self.node_feature_names) == len(self.node_dtypes)
            if str in self.node_dtypes:
                self.categorical_features['node'] += [self.node_feature_names[i] for i, j in enumerate(self.node_dtypes) if j == str]
        else:
            self.node_dtypes = None
        # edges
        if self.edge_feature_names:
            self.edge_dtypes = [type(dtype) for dtype in agraph.edge_features[0]]
            assert len(self.edge_feature_names) == len(self.edge_dtypes)
            if str in self.edge_dtypes:
                self.categorical_features['edge'] += [self.edge_feature_names[i] for i, j in enumerate(self.edge_dtypes) if j == str]
        else:
            self.edge_feature_names = None

    def get_graphs(self) -> list:
        return NotImplementedError

    def __len__(self) -> int:
        return len(self.graphs)


class ParsedGxlCxlDataset(ParsedGxlDataset):
    def __init__(self, path_to_dataset: str, categorical_features: dict,
                 ignore_coordinates: bool = True, center_coordinates: bool = False, subset: str = None):
        """
        This class creates a dataset object containing all the graphs parsed from gxl files as ParsedGxlGraph objects

        Parameters
        ----------
        ignore_coordinates : bool
            if true, coordinates are removed from the node features
        path_to_dataset: str
            path to the 'data' folder with gxl files
        categorical_features : dict
            optional parameter: dictionary with first level 'edge' and/or 'node' and then a list of the attribute
            names that should be one-hot encoded
            TODO: actually make this optional
            e.g. {'node': ['charge']}}
        subset: if specified, has to be either 'train', 'test' or 'val'
        """
        super(ParsedGxlCxlDataset, self).__init__(path_to_dataset=path_to_dataset,
                                                  ignore_coordinates=ignore_coordinates,
                                                  categorical_features=categorical_features)
        self.subset = subset
        # optional arguments
        # self.categorical_features = categorical_features
        self.center_coordinates = center_coordinates

        # if there are string-based features, we need to encode them as a one-hot vector
        # if self.node_feature_names and len(self.categorical_features['node']) > 0:
        #     self.one_hot_encode_nodes()
        # if self.edge_feature_names and len(self.categorical_features['edge']) > 0:
        #     self.one_hot_encode_edge_features()

    def get_graphs(self) -> list:
        """
        Create the graph objects. If self.subset is set only the specified subset is loaded,
        otherwise the whole dataset is loaded.

        Returns
        -------
        graphs: list [ParsedGxlGraph obj]
            list of graphs parsed from the gxl files
        """
        graphs = []
        for filename in self.all_file_names:
            try:
                try:
                    subset, class_label = self.filename_split_class[filename]
                    if self.subset:
                        assert subset == self.subset
                except KeyError:
                    logging.warning(f'{filename} does not appear in the dataset split files. File is skipped.')
                    continue

                g = ParsedGxlGraph(path_to_gxl=os.path.join(self.path_to_dataset, filename),
                                   subset=subset, class_label=self.class_int_encoding[class_label],
                                   remove_coordinates=self.remove_coordinates, center_coordinates=self.center_coordinates)
                graphs.append(g)
            except InvalidFileException:
                logging.warning(f'File {filename} is invalid. Please verify that the file contains the expected attributes '
                                f'(node, edge, id, edgeids and edgemode)')
                self.invalid_files.append(filename)

        return graphs

    @property
    def filename_split_class(self) -> dict:
        filename_split_class = {}
        if self.subset:
            for filename, class_label in self.dataset_split[self.subset].items():
                filename_split_class[filename] = (self.subset, class_label)
        else:
            for subset, d in self.dataset_split.items():
                for filename, class_label in d.items():
                    filename_split_class[filename] = (subset, class_label)
        return filename_split_class

    @property
    def class_names(self) -> list:
        if self.subset:
            class_labels = set([class_label for filename, class_label in self.dataset_split[self.subset].items()])
        else:
            class_labels = set([class_label for subset, d in self.dataset_split.items() for filename, class_label in d.items()])
        return sorted(class_labels)

    @property
    def class_int_encoding(self) -> dict:
        return {c: i for i, c in enumerate(self.class_names)}

    @property
    def all_file_names(self):
        if self.subset:
            filenames = [f for f in self.dataset_split[self.subset] if os.path.isfile(os.path.join(self.path_to_dataset, f)) if '.gxl' in f]
        else:
            filenames = [f for f in os.listdir(self.path_to_dataset) if os.path.isfile(os.path.join(self.path_to_dataset, f)) if '.gxl' in f]
        return filenames

    @property
    def dataset_split(self) -> dict:
        """
        Create a dictionary that contains the dataset split as well as the class labels

        Returns
        -------
        filename_class_split: dict
            {'train': {'file1.gxl': 'class_label' }, ...}
        """
        filename_class_split = {}

        for subset in ['train', 'test', 'val']:
            cxl_file = os.path.join(self.path_to_dataset, subset + '.cxl')
            if not os.path.isfile(os.path.join(self.path_to_dataset, subset + '.cxl')):
                logging.error(f'File {cxl_file} not found. Make sure file is called either train.cxl, val.cxl or test.cxl')
            tree = ET.parse(cxl_file)
            root = tree.getroot()
            filename_class_split[subset] = {i.attrib['file']: i.attrib['class'] for i in root.iter('print')}

        return filename_class_split


class ParsedGxlDatasetFolders(ParsedGxlDataset):
    def __init__(self, path_to_dataset: str, categorical_features: str, subset: str = '',
                 ignore_coordinates: bool = True, percent: float = None,
                 balanced_loading: bool = False):
        """
        This class creates a dataset object containing all the graphs parsed from gxl files as ParsedGxlGraph objects

        Parameters
        ----------
        ignore_coordinates : bool
            if true, coordinates are removed from the node features
        path_to_dataset: str
            path to the 'data' folder with gxl files
        subset: has to be either 'train', 'test' or 'val' (or '' if unspecified)
        # TODO: implement categorical feature loading
        """
        self.percent = percent
        self.balanced_loading = balanced_loading
        self.subset = subset
        super().__init__(path_to_dataset=os.path.join(path_to_dataset), ignore_coordinates=ignore_coordinates,
                         categorical_features=categorical_features)

    @property
    def class_filename_dict(self):
        class_filename_dict = {label: [] for label in np.unique([int(os.path.basename(os.path.dirname(filepath))) for filepath in self.all_file_paths])}
        for filepath in self.all_file_paths:
            class_label = int(os.path.basename(os.path.dirname(filepath)))
            class_filename_dict[class_label].append(filepath)

        # sample subset of each class if percentage is smaller than 1
        if self.percent:
            class_filename_dict = {cl: random.sample(class_filename_dict[cl], max(1, int(self.percent*len(path_list))))
                                   for cl, path_list in class_filename_dict.items()}
        return class_filename_dict

    def get_graphs(self) -> list:
        """
        Create the graph objects. If self.subset is set only the specified subset is loaded,
        otherwise the whole dataset is loaded.

        Returns
        -------
        graphs: list [ParsedGxlGraph obj]
            list of graphs parsed from the gxl files
        """
        graphs = []
        # TODO: add option for non integer class folder names?
        class_sizes = {class_label: len(filepath_list) for class_label, filepath_list in self.class_filename_dict.items()}
        max_class_size = max(class_sizes.values())
        for class_label, filepath_list in self.class_filename_dict.items():
            # up-sample smaller classes to largest class
            if self.balanced_loading and len(filepath_list) < max_class_size:
                multiplier = max_class_size // len(filepath_list)
                filepath_list_upsampled = filepath_list * multiplier
                filepath_list = filepath_list_upsampled + \
                                random.sample(filepath_list, max_class_size - len(filepath_list_upsampled))

            for filepath in filepath_list:
                try:
                    g = ParsedGxlGraph(path_to_gxl=filepath, subset=self.subset, class_label=int(class_label),  # class label has to be cast to int from int64 for json saving later
                                       remove_coordinates=self.remove_coordinates)
                    graphs.append(g)
                except InvalidFileException:
                    logging.warning(f'File {filepath} is invalid. Please verify that the file contains the expected attributes '
                                    f'(node, edge, id, edgeids and edgemode)')
                    self.invalid_files.append(filepath)
        return graphs

    @property
    def all_file_paths(self):
        if not os.path.isdir(os.path.join(self.path_to_dataset), self.subset):
            logging.error(f'The specified input directory {self.path_to_dataset} does not contain the subset folder {self.subset}')
            sys.exit(-1)
        filenames = glob.glob(os.path.join(self.path_to_dataset, '**/*.gxl'), recursive=True)
        return filenames


class ParsedGxlSplitJson(ParsedGxlDataset):
    def __init__(self, path_to_dataset: str, categorical_features: dict, split_json: str,
                 ignore_coordinates: bool = True):
        """
        This class creates a dataset object containing all the graphs parsed from gxl files as ParsedGxlGraph objects

        Parameters
        ----------
        ignore_coordinates : bool
            if true, coordinates are removed from the node features
        path_to_dataset: str
            path to the 'data' folder with gxl files
        # TODO: implement categorical feature loading
        """
        self.split_json = split_json
        super().__init__(path_to_dataset=os.path.join(path_to_dataset),
                         ignore_coordinates=ignore_coordinates, categorical_features=categorical_features)

    def get_graphs(self) -> list:
        """
        Create the graph objects. If self.subset is set only the specified subset is loaded,
        otherwise the whole dataset is loaded.

        Returns
        -------
        graphs: list [ParsedGxlGraph obj]
            list of graphs parsed from the gxl files
        """
        graphs = []
        for subset, subset_dict in self.split_json.items():
            for class_label, filename_list in subset_dict.items():
                for filename in filename_list:
                    try:
                        g = ParsedGxlGraph(path_to_gxl=os.path.join(self.path_to_dataset, f'{filename}.gxl'),
                                           subset=subset, class_label=int(class_label),  # class label has to be cast to int from int64 for json saving later
                                           remove_coordinates=self.remove_coordinates)
                        graphs.append(g)
                    except InvalidFileException:
                        logging.warning(f"File {os.path.join(self.path_to_dataset, f'{filename}'.gxl)} is invalid. "
                                        f"Please verify that the file contains the expected attributes "
                                        f"(node, edge, id, edgeids and edgemode)")
                        self.invalid_files.append(filename_list)
        return graphs

    @property
    def all_file_paths(self):
        if not os.path.isdir(self.path_to_dataset):
            logging.error(f'The specified input directory {self.path_to_dataset} does not exist')
            sys.exit(-1)
        filenames = glob.glob(os.path.join(self.path_to_dataset, '*.gxl'))
        return filenames


class MultiParsedGxlDatasets:
    def __init__(self, dataset_list: list):
        attributes = self.get_attributes(dataset_list)
        assert set().union(*[set(a) for a in attributes]) == set(attributes[0])

        # combine the attributes
        self.comb_attributes = {k: None for k in set(attributes[0])}
        self.set_comb_attrib(dataset_list)
        self.set_class_attrib()

    def get_attributes(self, dataset_list):
        a = [[i for i in d if '__' not in i] for d in [dir(d) for d in dataset_list]]
        return [[attrib for attrib in attribs if not callable(getattr(obj, attrib))] for attribs, obj in zip(a, dataset_list)]

    def set_comb_attrib(self, dataset_list):
        for i in range(len(dataset_list)-1):
            ds1 = dataset_list[i]
            ds2 = dataset_list[i+1]
            for key in self.comb_attributes.keys():
                if getattr(ds1, key) == getattr(ds2, key):
                    self.comb_attributes[key] = getattr(ds2, key)
                else:
                    if self.comb_attributes[key] is None:
                        self.comb_attributes[key] = self._update_comb_attrib(ds1, ds2, key)
                    else:
                        self.comb_attributes[key] = self.comb_attributes[key] + self._update_comb_attrib(ds1, ds2, key)

    @staticmethod
    def _update_comb_attrib(ds1, ds2, key):
        if isinstance(getattr(ds1, key), list):
            return getattr(ds1, key) + getattr(ds2, key)
        else:
            return [getattr(ds1, key), getattr(ds2, key)]

    def set_class_attrib(self):
        # set the attributes
        for key, val in self.comb_attributes.items():
            if 'subset' in key and isinstance(val, list):
                val = '-'.join(val)
            if key == 'config' and isinstance(val, list):
                # TODO: make this more general and less hacky
                val = val[0]
                val['dataset_split'] = '-'.join(self.comb_attributes['subset'])
            setattr(self, key, val)

