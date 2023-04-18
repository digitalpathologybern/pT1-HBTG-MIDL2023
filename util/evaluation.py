import pandas as pd
import numpy as np
import torch
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance


def pretty_print_performance(performance):
    # check if we have multiple metrics
    p = {}
    for metric, vals in performance.items():
        if vals.ndim > 0:
            for i, v in enumerate(vals):
                p[f'{metric}_{i}'] = v
        else:
            p[metric] = vals
    return p


def prepare_average_precision(outputs, top_k=None, return_all=False):
    """
    reformats the predicitons for keyword spotting so that it can be used to calculate the retrieval mAP
    :param return_all: default False. If True, the complete results list is returned as well as the topk
    :param outputs: batch output of network and the label for each element in the batch
    :param top_k: default is None (=full).
                  auto: sets it so only top-k elements are considered, where k is the total number of matches possible for this class.
                  k: sets it so only top-k elements are considered (k can be set to any number)
                  auto-k: sets it so only top-i elements are considered, where i is the total number of matches possible for this class, but a minimum of k.
    """
    pred = torch.cat([o[0] for o in outputs], axis=0)
    labels = torch.cat([o[1] for o in outputs])
    # evaluate and log
    dist_matrix = pairwise_cosine_similarity(pred)
    y_true_topk_all = []
    y_score_topk_all = []
    index_topk_all = []
    if return_all:
        y_true_all = []
        y_score_all = []
        index_all = []
    for ind, label in enumerate(labels):
        # check if batch is not filled
        row = dist_matrix[ind][:len(labels)]
        # remove the diagonal and the label entry
        distances = torch.cat((row[:ind], row[ind+1:]))
        row_labels = torch.cat((labels[:ind], labels[ind+1:]))
        # create df to then sort by distance
        df = torch.stack((row_labels, distances))
        df = df[:, df[1].sort(descending=True)[1]]
        # compare to gt
        y_score = label == df[0]
        # shorten if necessary
        if top_k is None:
            y_score_topk = y_score
        elif top_k == 'auto':
            y_score_topk = y_score[:sum(y_score)]
        elif type(top_k) == int:
            y_score_topk = y_score[:top_k]
        elif 'auto-' in top_k:
            k = int(top_k.split('-')[1])
            y_score_topk = y_score[:k] if k > sum(y_score) else y_score[:sum(y_score)]
        else:
            print("Invalid setting for top-k, computing over full array.")
            y_score_topk = y_score

        y_score_topk_all.append([i.to(torch.bool) for i in y_score_topk])
        y_true_topk_all.append(sorted(y_score_topk, reverse=True))
        index_topk_all.append([ind] * len(y_score_topk))

        if return_all:
            y_score_all.append([i.to(torch.bool) for i in y_score])
            y_true_all.append(sorted(y_score, reverse=True))
            index_all.append([ind] * len(y_score))

    y_true_topk_all = torch.tensor([item for sublist in y_true_topk_all for item in sublist])
    y_score_topk_all = torch.tensor([item for sublist in y_score_topk_all for item in sublist],
                                    dtype=torch.float)
    index_topk_all = torch.tensor([item for sublist in index_topk_all for item in sublist])

    if return_all:
        y_true_all = torch.tensor([item for sublist in y_true_all for item in sublist])
        y_score_all = torch.tensor([item for sublist in y_score_all for item in sublist], dtype=torch.float)
        index_all = torch.tensor([item for sublist in index_all for item in sublist])
        return (y_score_all, y_true_all, index_all), (y_score_topk_all, y_true_topk_all, index_topk_all)
    else:
        return (y_score_topk_all, y_true_topk_all, index_topk_all)

