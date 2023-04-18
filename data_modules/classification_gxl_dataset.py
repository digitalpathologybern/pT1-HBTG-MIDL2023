import os
import numpy as np
import torch
import shutil
import json
import logging
import pandas as pd
import sys
from sklearn.preprocessing import OneHotEncoder

from torch_geometric.data import InMemoryDataset, Data
from data_modules.util.gxl_classification_parsers import ParsedGxlCxlDataset, ParsedGxlDatasetFolders, \
    ParsedGxlSplitJson


class GxlDataset(InMemoryDataset):
    def __init__(self, input_folder, cxl: bool = False, transform=None, pre_transform=None,
                 categorical_features_json: str = None, split_json: str = None, subset: str = '',
                 ignore_coordinates: bool = True, percent: float = None, add_mlp_feature_csv: str = None,
                 balanced_loading: bool = False, **kwargs):
        """
        This class reads a dataset in gxl format

        Parameters
        ----------
        use_position
        input_folder: str
            Path to the dataset folder. There has to be a sub-folder 'data' where the graph gxl files and the train.cxl,
            val.cxl and test.cxl files are.
        transform:
        pre_transform:
        categorical_features : str
            path to json file
            optional parameter: dictionary with first level 'edge' and/or 'node' and then a list of the attribute names
            that need to be one-hot encoded ( = are categorical)
            e.g. {'node': ['symbol'], 'edge': ['valence']}}
        rebuild_dataset: bool
            True if dataset should be re-processed (deletes and re-generates the processes folder).
        subset: str / list of str
            name or list of the subfolders / corresponding cxl file names to load (default is None)
        balanced_loading: bool
            default True. if set to false, no oversampling is done to balance the classes
        """
        self.root = input_folder
        self.cxl = cxl
        self.subset = subset
        self.split_json = split_json
        self.categorical_features = categorical_features_json  # only implemented for cxl datasets
        self.use_position = ignore_coordinates
        self.percent = percent  # only implemented for folder dataset
        self.add_mlp_features = self.read_add_mlp_feature_csv(add_mlp_feature_csv)
        # TODO: improve this? Add a post upsampling?
        self.balanced_loading = balanced_loading  # only implemented for folder dataset
        self.name = os.path.basename(input_folder)

        super(GxlDataset, self).__init__(self.root, transform, pre_transform)


        # split the dataset and the slices into three subsets
        self.data, self.slices, self.config = torch.load(self.processed_paths[0])
        # check if mlp features are present in graphs if add_mlp_features are provided

        # override the categorical features
        self._categorical_features = self.config["categorical_features"]
        # set the number of node features
        self.num_node_features = len(self.config['node_feature_names'])

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the dataset."""
        return self._num_node_features

    @num_node_features.setter
    def num_node_features(self, num_node):
        if num_node is None:
            num_node = 0
        self._num_node_features = num_node

    @property
    def categorical_features(self):
        return self._categorical_features

    @categorical_features.setter
    def categorical_features(self, categorical_features_json):
        """
        dictionary with first level 'edge' and/or 'node' and then a list of the attribute names that should be one-hot
        encoded, e.g. {'node': ['charge'], 'edge': ['valence]}.

        Parameters
        ----------
        categorical_features_json: str
            path to the json file
        """
        if categorical_features_json is None:
            categorical_features = {'node': [], 'edge': []}
        else:
            # read the json file
            logging.info('Loading categorical variables from JSON ({})'.format(categorical_features_json))
            with open('strings.json') as f:
                categorical_features = json.load(categorical_features_json)
            # add missing keys
            if 'edge' not in categorical_features:
                categorical_features['edge'] = []
            if 'node' not in categorical_features:
                categorical_features['node'] = []
        self._categorical_features = categorical_features

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root_path):
        if not os.path.isdir(root_path):
            logging.error(f'Folder {root_path} does not exist.')
            sys.exit(-1)
        self._root = root_path

    @property
    def raw_file_names(self):
        """
        The name of the files to find in the :obj:`self.raw_dir` folder in order to skip the download.
        Empty list because files cannot be downloaded automatically.
        """
        return []

    @property
    def processed_file_names(self):
        processed_path = os.path.join(self.root, 'processed')
        # check if /processed folder exists, if not create it
        if not os.path.exists(processed_path):
            os.mkdir(processed_path)
        filler = '-' if len(self.subset) > 0 else ''
        return [os.path.basename(self.root) + filler + self.subset + '.dataset']

    def read_add_mlp_feature_csv(self, csv_path):
        # TODO: invasion depth gets one hot encoded, why? FIX!!!!
        if csv_path is None:
            return csv_path
        df = pd.read_csv(csv_path, index_col=0)
        df.replace('na', np.NaN, inplace=True)
        df.replace('N.A.', np.NaN, inplace=True)
        df.replace('NA', np.NaN, inplace=True)
        # one hot encode categorical features
        one_hot = pd.get_dummies(df)
        return one_hot

    def get_gxl_dataset(self):
        # gxl dataset which uses cxl files to split the data and identify the classes (e.g. IAMDB datasets)
        if self.cxl:
            gxl_dataset = ParsedGxlCxlDataset(path_to_dataset=os.path.join(self.root, 'data'),
                                              categorical_features=self.categorical_features, subset=self.subset,
                                              ignore_coordinates=self.use_position)
        # gxl dataset which has a json file to split the data and identify the classes (either for CV or train/val/test)
        elif self.split_json is not None:
            gxl_dataset = ParsedGxlSplitJson(path_to_dataset=self.root,
                                             ignore_coordinates=self.use_position,
                                             categorical_features=self.categorical_features,
                                             split_json=self.split_json)

        # gxl files are already split into the subset with folders, and each subset folder contains one data folder
        # per class (e.g. train/0/, train/1/, ...)
        else:
            gxl_dataset = ParsedGxlDatasetFolders(path_to_dataset=os.path.join(self.root, self.subset),
                                                  subset=self.subset, ignore_coordinates=self.use_position,
                                                  percent=self.percent,
                                                  categorical_features=self.categorical_features,
                                                  balanced_loading=self.balanced_loading)
            # else:
            #
            # gxl_datasets = [ParsedGxlDatasetFolders(path_to_dataset=os.path.join(self.root, s),
            #                                         subset=s, ignore_coordinates=self.use_position,
            #                                         percent=self.percent,
            #                                         categorical_features=self.categorical_features,
            #                                         balanced_loading=self.balanced_loading) for s in
            #                 self.subset]
            # gxl_dataset = MultiParsedGxlDatasets(gxl_datasets)
            # self.subset = gxl_dataset.subset

        return gxl_dataset

    def download(self):
        """
        Files cannot be automatically downloaded.
        """
        pass

    def process(self):
        """
        Processes the dataset to the :obj:`self.processed_dir` folder.
        """
        # # create the dataset
        gxl_dataset = self.get_gxl_dataset()

        # make a csv with the number of nodes per graph
        if not os.path.isfile(os.path.join(os.path.dirname(self.processed_paths[0]), 'nb_nodes_edges_per_graph.csv')):
            df = pd.DataFrame.from_records(
                [[g.filename, g.class_label, int(g.nb_of_nodes), int(g.nb_of_edges)] for g in gxl_dataset.graphs],
                columns=['filename', 'class_label', 'nb_of_nodes', 'nb_of_edges']).sort_values(by=['filename'])
            df.to_csv(
                os.path.join(os.path.dirname(self.processed_paths[0]), f'{self.subset}_nb_nodes_edges_per_graph.csv'),
                index=False)

        config = gxl_dataset.config
        data_list = []

        # create the dataset lists: transform the graphs in the GxlDataset into pytorch geometric Data objects
        file_names = gxl_dataset.file_names
        # save the file_names list
        if not os.path.isfile(
                os.path.join(os.path.dirname(self.processed_paths[0]), f'{self.subset}_file_name_list.csv')):
            pd.DataFrame({'file_names': file_names}).to_csv(
                os.path.join(os.path.dirname(self.processed_paths[0]), f'{self.subset}_file_name_list.csv'),
                index=False)

        # TODO: add pre-transform
        for graph in gxl_dataset.graphs:
            # x (Tensor): Node feature matrix with shape :obj:`[num_nodes, num_node_features]`
            # y (Tensor): Graph or node targets with arbitrary shape.
            # edge_index (LongTensor): Graph connectivity in COO format with shape :obj:`[2, num_edges]`
            # edge_attr (Tensor): Edge feature matrix with shape :obj:`[num_edges, num_edge_features]`
            # y (Tensor): Graph or node targets with arbitrary shape

            # node
            x = torch.tensor(graph.node_features, dtype=torch.float)
            pos = torch.tensor(graph.node_positions, dtype=torch.float)
            # edges
            edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(graph.edge_features, dtype=torch.float)
            # make graph undirected if necessary
            if gxl_dataset.edge_mode == 'undirected' and len(edge_index) == 2:
                row, col = edge_index
                new_row = torch.cat([row, col], dim=0)
                new_col = torch.cat([col, row], dim=0)
                edge_index = torch.stack([new_row, new_col], dim=0)
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
            # labels (cannot be a string!)
            y = graph.class_label
            # add additional features for MLP if available
            if self.add_mlp_features is not None:
                file_id = graph.file_id.split('_hotspot')[0]  # necessary to split off the hotspot ID
                add_feat = torch.unsqueeze(torch.tensor(self.add_mlp_features.loc[file_id].to_list()), 0)
            else:
                add_feat = None
            # file names
            file_name_ind = file_names.index(graph.filename)
            # make the graph
            g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y, file_name_ind=file_name_ind,
                     add_mlp_features=add_feat, subset=graph.subset)
            data_list.append(g)

        # save the data
        data, slices = self.collate(data_list)

        if data.y is not None:
            # add config calculations
            index, counts = np.unique(np.array(data.y), return_counts=True)
            counts = list(counts / sum(counts))
            # convert from numpy for json
            config['class_freq'] = ([int(i) for i in index], [float(i) for i in counts])
            config['file_names'] = file_names
        if self.add_mlp_features is not None:
            config['csv_mlp_feature_names'] = self.add_mlp_features.T.to_dict('list')

        # save the config
        if not os.path.isfile(
                os.path.join(os.path.dirname(self.processed_paths[0]), f'{self.subset}_graphs_config.json')):
            with open(os.path.join(os.path.dirname(self.processed_paths[0]), f'{self.subset}_graphs_config.json'),
                      'w') as fp:
                json.dump(config, fp, indent=4)

        torch.save((data, slices, config), self.processed_paths[0])
