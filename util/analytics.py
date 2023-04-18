import numpy as np


# Analytics handling
def get_node_feature_names(dataset):
    return list(range(dataset.num_node_features))


def get_edge_feature_names(dataset):
    return list(range(dataset.num_edge_features))


def get_class_names(dataset):
    return list(range(int(dataset.num_classes)))


def get_class_freq(dataset):
    try:
        # this only works for the GXL datasets
        class_frequencies = np.array(dataset.config['class_freq'][1])
    except AttributeError:
            labels = [i.y.numpy()[0] for i in dataset]
            class_frequencies = np.unique(labels, return_counts=True)[1] / len(labels)
    return class_frequencies
