# Utils
import argparse
from data_modules.util.gxl_classification_parsers import ParsedGxlDatasetFolders
import pandas as pd
import re
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx

def get_stats_gxl_class_folders(input_folder, output_folder):
    if output_folder is None:
        output_folder = input_folder

    dataset = ParsedGxlDatasetFolders(path_to_dataset=os.path.join(input_folder, 'train'), subset='train')

    class_names = {class_id: re.search(r'_\d*(.*)_', file_list[0]).group(1) for class_id, file_list in dataset.class_filename_dict.items()}
    df = pd.DataFrame.from_dict({'class labels': sorted(class_names.keys()), 'class names': [class_names[k] for k in sorted(class_names.keys())]})

    # graph stats
    graphs_per_class = {i: [] for i in dataset.class_names}
    radii_per_class = {i: [] for i in dataset.class_names}
    diameter_per_class = {i: [] for i in dataset.class_names}
    nb_nodes_per_class = {i: [] for i in dataset.class_names}
    nb_edges_per_class = {i: [] for i in dataset.class_names}
    # convert graph to networkx graph
    for graph in dataset.graphs:
        label = graph.class_label
        nb_nodes_per_class[label] = len(graph.node_features)
        nb_edges_per_class[label] = len(graph.edges)

        # node
        x = torch.tensor(graph.node_features, dtype=torch.float)
        pos = torch.tensor(graph.node_positions, dtype=torch.float)
        # edges
        edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(graph.edge_features, dtype=torch.float)
        row, col = edge_index
        new_row = torch.cat([row, col], dim=0)
        new_col = torch.cat([col, row], dim=0)
        edge_index = torch.stack([new_row, new_col], dim=0)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        # labels (cannot be a string!)
        y = graph.class_label
        # make the graph
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)
        g_nx = to_networkx(g, to_undirected=True)
        radii_per_class[label] = networkx.radius(g_nx)
        diameter_per_class[label] = networkx.diameter(g_nx)

    print('vlakjd')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile the CSV results from wandb into a nice exel and figures.')
    parser.add_argument('--data-folder', type=str, help='path to dataset')
    parser.add_argument('--dataset-type', type=str, help='type of the dataset (only implemented for IAMBD gxl so far)')
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path where outputs should be save to (default is same as data folder).')
    args = parser.parse_args()

    categorical_features = {'node': [], 'edge': []}

    if args.dataset_type == 'gxl-folder':
        get_stats_gxl_class_folders(input_folder=args.data_folder, output_folder=args.output_folder)

