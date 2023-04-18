import glob
import os
import json
import pandas as pd
import numpy as np
from torchmetrics import MetricCollection
import torch
from torchmetrics.classification import BinaryRecall, BinaryF1Score, BinaryPrecision, BinaryAccuracy

from util_scripts.aggregate_results_csvs import compile_summary_table


# %%
# for i in `ssh username@host 'find path/to/results -name "*summary.csv"'`; do scp "username@host:$i" . ; done


def get_pid(masterfile_path, cv_split_json):
    patient_data_all = pd.read_excel(masterfile_path, sheet_name='combined')
    patient_data = patient_data_all.loc[patient_data_all['In HBTG dataset'], :]
    # patient_data.dropna(subset=["CD8 ID", "HE ID"], inplace=True)

    with open(cv_split_json) as f:
        data = json.load(f)

    all_patients_cv = []
    for cv_ind, cv_dict in data.items():
        for label, file_list in cv_dict.items():
            all_patients_cv = all_patients_cv + file_list

    pid_per_cv = {}
    for cv_ind, cv_dict in data.items():
        for label, file_list in cv_dict.items():
            all_patients_cv = all_patients_cv + file_list
            for f in file_list:
                pid_per_cv[f] = [cv_ind, label, patient_data[patient_data['CD8 filename'] == f]['Patient-Nr'].values[0]]

    pid_per_cv = pd.DataFrame.from_dict(pid_per_cv, columns=['cv_ind', 'label', 'pid'], orient='index')
    return pid_per_cv


# evaluate per PID not per slide
def eval_per_pid(df_softmax, cutoff=0.5, majority_vote=False):
    # df_softmax.sort_values(by=['pid'], inplace=True)
    perf = {}
    for cv_ind in range(max(df_softmax['cv-fold']) + 1):
        df_cv = df_softmax[df_softmax['cv-fold'] == cv_ind]
        perf_cv = eval_per_cv(df_cv=df_cv, cutoff=cutoff, vote=majority_vote)
        perf[f'cv{cv_ind}'] = perf_cv

    perf = pd.DataFrame.from_dict(perf)
    perf['mean'] = perf.mean(axis=1)
    perf['std'] = perf.std(axis=1)
    return perf


# evaluate per PID not per slide
def classif_per_pid_ensemble(df_softmax):
    df_agg = pd.concat([df_softmax.filter(regex=f"softmax_class0").iloc[:, 0:5].mean(axis=1),
                        df_softmax.filter(regex=f"softmax_class1").iloc[:, 0:5].mean(axis=1)],
                       axis=1, keys=['avg_class0', 'avg_class1'])
    df_agg = aggregate_per_pid(df_agg, df_softmax['pid'])
    classif = 0.5 >= df_agg.avg_class0
    classif = classif.astype(int)
    return classif


def eval_per_cv(df_cv, cutoff=0.5, vote=False):
    metrics_cv = {}
    if not vote:
        for run_id in range(len([i for i in df_cv.columns if 'r' in i]) // 2):
            df_run = df_cv.filter(regex=f"r{run_id}")
            df_run = aggregate_per_pid(df_run, df_cv['pid'])
            target = df_cv.set_index('pid')['class_label']
            target = target[~target.index.duplicated()]
            metrics_run = get_run_metrics(df_run=df_run, target=target, cutoff=cutoff)
            metrics_cv[f'r{run_id}'] = pd.Series(metrics_run)
        return pd.DataFrame.from_dict(metrics_cv).mean(axis=1)
    else:
        # average the softmax
        df_agg = pd.concat([df_cv.filter(regex=f"softmax_class0").iloc[:, 0:5].mean(axis=1),
                            df_cv.filter(regex=f"softmax_class1").iloc[:, 0:5].mean(axis=1)],
                           axis=1, keys=['avg_class0', 'avg_class1'])
        # df_agg = pd.concat([df_cv.filter(regex=f"softmax_class0").iloc[:, 0:5].mean(axis=1),
        #                     df_cv.filter(regex=f"softmax_class1").iloc[:, 0:5].mean(axis=1)],
        #                    axis=1, keys=['avg_class0', 'avg_class1'])
        df_agg = aggregate_per_pid(df_agg, df_cv['pid'])
        target = df_cv.set_index('pid')['class_label']
        target = target[~target.index.duplicated()]
        return pd.Series(get_run_metrics(df_run=df_agg, target=target, cutoff=cutoff), name='avg_vote')


def get_run_metrics(df_run, target, cutoff=0.5):
    results = {}
    # calculate for class 1
    metrics_c1 = MetricCollection(dict(F1Score=BinaryF1Score(threshold=cutoff),
                                       Precision=BinaryPrecision(threshold=cutoff),
                                       Recall=BinaryRecall(threshold=cutoff),
                                       Accuracy=BinaryAccuracy(threshold=cutoff),
                                       ))
    metrics_c1(torch.tensor(df_run.values[:, 1]), torch.tensor(target.values))
    results_c1 = metrics_c1.compute()
    # calculate for class 0
    cutoff_0 = 1 - cutoff
    metrics_c0 = MetricCollection(dict(F1Score=BinaryF1Score(threshold=cutoff_0),
                                       Precision=BinaryPrecision(threshold=cutoff_0),
                                       Recall=BinaryRecall(threshold=cutoff_0),
                                       Accuracy=BinaryAccuracy(threshold=cutoff_0),
                                       ))
    # switch positive class to class0
    metrics_c0(torch.tensor(df_run.values[:, 0]), torch.tensor(1 - target.values))
    results_c0 = metrics_c0.compute()
    # get averages
    for metric in sorted(results_c0.keys()):
        results[f'{metric}'] = float((results_c0[metric] + results_c1[metric]) / 2)
        results[f'{metric}_c0'] = float(results_c0[metric])
        results[f'{metric}_c1'] = float(results_c1[metric])

    return results


def pretty_print_metrics(metrics_dict):
    performance_flat = {}
    for m, val in metrics_dict.items():
        if val.dim() > 0:
            for i in range(len(val)):
                performance_flat[f'{m}{i}'] = float(val[i])
        else:
            performance_flat[m] = float(val)
    return performance_flat


def aggregate_per_pid(df, pid):
    multi = set([x for x in pid.values if pid.to_list().count(x) > 1])
    df_pid = df.set_index(pid)
    df_pid_single = df_pid[~df_pid.index.duplicated(keep='first')]
    for i in multi:
        rows = df_pid.loc[[i]]
        rows.reset_index(inplace=True, drop=True)
        ind = abs(rows[rows.columns[0]] - 0.5).idxmax()
        df_pid_single.loc[[i]] = rows.loc[[ind]].values
    return df_pid_single


def compute_per_pid(csv_files, masterfile_path, cv_split_json, out_folder, cutoff=0.5, majority_vote=False, nbruns=None):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    pid_per_cv = get_pid(masterfile_path=masterfile_path, cv_split_json=cv_split_json)
    csv_dfs = {os.path.basename(f).split('-softmax')[0]: pd.read_csv(f, index_col=0) for f in csv_files}
    # one_df = csv_dfs['graphsage_jk-tb_to_tb_delaunay-l_to_tb_radius100']
    for exp_name, df in csv_dfs.items():
        if nbruns is not None:
            df = df.iloc[:, :df.columns.tolist().index(f'r{nbruns-1}_softmax_class0')+2]
        df = pd.concat([df, pid_per_cv['pid']], axis=1)
        results = eval_per_pid(df, cutoff=cutoff, majority_vote=majority_vote)
        results.to_csv(os.path.join(out_folder, f'{exp_name}-summary.csv'))


def process(subfolder, ensemble, nbruns=None):
    results_folder = f'/Users/linda/Nextcloud/PhD/Papers and Conferences/2023-07 MIDL/results-midl/softmax/{subfolder}'
    csv_files = glob.glob(os.path.join(results_folder, '*.csv'))
    # csv_files = glob.glob(os.path.join(results_folder, '*dino200*.csv'))
    # cutoff = 0.35

    masterfile_path = "/Users/linda/Nextcloud/PhD/Projects/Rising Tide/data pT1/patient data pT1/pT1-Masterfile-full-NC.xlsx"
    cv_split_json = '/Users/linda/Documents/PhD-local/BTS/pT1/manual-hotspots/graphs/pT1-Masterfile-full-NC-cv5.json'
    if ensemble:
        ensemble_tag = f'aggr{nbruns}-' if ensemble else ''
        out_folder = f'/Users/linda/Nextcloud/PhD/Papers and Conferences/2023-07 MIDL/results-midl/pid/{ensemble_tag}{subfolder}'
    else:
        out_folder = f'/Users/linda/Nextcloud/PhD/Papers and Conferences/2023-07 MIDL/results-midl/pid/no_aggr-{subfolder}'
    compute_per_pid(csv_files=csv_files, masterfile_path=masterfile_path, cv_split_json=cv_split_json,
                    out_folder=out_folder, majority_vote=ensemble, nbruns=nbruns)

    compile_summary_table(out_folder, tag=f'{os.path.basename(out_folder)}')


# process(subfolder='xytype', ensemble=True, nbruns=5)
process(subfolder='dinotypecoord', ensemble=False)

#%% process a list of folders to get the metrics
# for folder in ['dinotype', 'dinotypecoord', 'type', 'xytype']:
#     process(subfolder=folder, ensemble=True)


#%% get the classification per patient

def get_classifications(subfolder):
    results_folder = f'/Users/linda/Nextcloud/PhD/Papers and Conferences/2023-07 MIDL/results-midl/softmax/{subfolder}'
    csv_files = glob.glob(os.path.join(results_folder, '*.csv'))

    pid_per_cv = get_pid(masterfile_path=masterfile_path, cv_split_json=cv_split_json)
    csv_dfs = {os.path.basename(f).split('-softmax')[0]: pd.read_csv(f, index_col=0) for f in sorted(csv_files)}
    # one_df = csv_dfs['graphsage_jk-tb_to_tb_delaunay-l_to_tb_radius100']
    results_all = {}
    for exp_name, df in csv_dfs.items():
        df = pd.concat([df, pid_per_cv['pid']], axis=1)
        df = aggregate_per_pid(df, pid_per_cv['pid'])
        classif = classif_per_pid_ensemble(df)
        results_all[exp_name] = classif.rename('pred_class')

    df_all = pd.DataFrame.from_dict(results_all)
    return df_all

# masterfile_path = "/Users/linda/Nextcloud/PhD/Projects/Rising Tide/data pT1/patient data pT1/pT1-Masterfile-full-NC.xlsx"
# cv_split_json = '/Users/linda/Documents/PhD-local/BTS/pT1/manual-hotspots/graphs/pT1-Masterfile-full-NC-cv5.json'
# subfolders = ['type', 'xytype', 'dinotypecoord', 'dinotype']
# df_all = [get_classifications(subfolder=s) for s in subfolders]
#
# pid_per_cv = get_pid(masterfile_path=masterfile_path, cv_split_json=cv_split_json).astype(int)
# pid_per_cv['CD8'] = pid_per_cv.index
# pid_per_cv.set_index('pid', inplace=True)
#
# #%%
# writer = pd.ExcelWriter('/Users/linda/Nextcloud/PhD/Papers and Conferences/2023-07 MIDL/results-midl/classifications.xlsx',
#                         engine='xlsxwriter')
# for s, df in zip(subfolders, df_all):
#     df_out = pd.concat([pid_per_cv, df], axis=1)
#     df_out.to_excel(writer, sheet_name=s)
# writer.close()

# %% get a list often misclassified graphs
#
#
# def correct_class(row, ref_df):
#     return row == ref_df[row.name]
#
#
# def get_misclassified_names(df):
#     softmax_c0 = df.filter(regex="_class0") < 0.5
#     df_class = softmax_c0.astype(int)
#     # df_label_class = df['class_label'].to_frame().join(df_class)
#     correct_class_df = df_class.apply(lambda row: correct_class(row, df['class_label']), axis=1)
#     less_3_correct = correct_class_df.sum(axis=1) < int(len(softmax_c0.columns)*0.6)
#     less_3_correct = correct_class_df.index[less_3_correct].to_list()
#     return less_3_correct
#
# def get_correctclassified_names(df):
#     softmax_c0 = df.filter(regex="_class0") < 0.5
#     df_class = softmax_c0.astype(int)
#     # df_label_class = df['class_label'].to_frame().join(df_class)
#     correct_class_df = df_class.apply(lambda row: correct_class(row, df['class_label']), axis=1)
#     less_3_correct = correct_class_df.sum(axis=1) >= int(len(softmax_c0.columns)*0.6)
#     less_3_correct = correct_class_df.index[less_3_correct].to_list()
#     return less_3_correct
#
#
# difficult_graphs = {exp_name: get_misclassified_names(csv_dfs[exp_name]) for exp_name in best_expnames}
# easy_graphs = {exp_name: get_correctclassified_names(csv_dfs[exp_name]) for exp_name in best_expnames}
#
# missed_by_all = np.unique([item for sublist in difficult_graphs.values() for item in sublist], return_counts=True)
# missed_by_all = missed_by_all[0][missed_by_all[1] == len(difficult_graphs.keys())]
#
# correct_by_all = np.unique([item for sublist in easy_graphs.values() for item in sublist], return_counts=True)
# correct_by_all = correct_by_all[0][correct_by_all[1] == len(easy_graphs.keys())]
