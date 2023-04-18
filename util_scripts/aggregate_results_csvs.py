import glob
import os
import re
import pandas as pd
import numpy as np


# for i in `ssh username@host 'find path/to/results -name "*summary.csv"'`; do scp "username@host:$i" . ; done


def compile_summary_table(results_folder, tag=None):
    if tag is None:
        tag = os.path.basename(results_folder)

    csv_dfs = {os.path.basename(f).split('-summary')[0]: pd.read_csv(f, index_col=0) for f in glob.glob(os.path.join(results_folder, '*.csv'))}

    dfs_mean_sd = {name: csv_dfs[name][['mean', 'std']]*100 for name in sorted(csv_dfs.keys())}

    graph_types = ['tb_to_tb_delaunay-l_to_tb_radius100', 'other_edge_fct_delaunay', 'other_edge_fct_hierarchical-cutoff-100']
    dfs_per_graph_type = {t: [] for t in graph_types}

    for exp_name, df in dfs_mean_sd.items():
        df = df.drop(['test/loss', 'test/Specificity'], errors='ignore')
        df = df.rename(index={i: i.replace(r'test/', '') for i in df.index.tolist()})
        g_type = [i for i in graph_types if i in exp_name].pop()
        df = pd.DataFrame(df.unstack().swaplevel().sort_index(), columns=[exp_name.replace(f'-{g_type}', '')])
        dfs_per_graph_type[g_type].append(df.T.sort_index())

    dfs_graphtype = {g_type: pd.concat(l) for g_type, l in dfs_per_graph_type.items()}

    iterables = [list(dfs_graphtype.keys()), list(dfs_graphtype[list(dfs_graphtype.keys())[0]].index)]
    multi_index = pd.MultiIndex.from_product(iterables, names=["graph type", "GNN"])
    df_meansd = pd.concat(dfs_graphtype.values())
    df_meansd.index = multi_index

    with pd.ExcelWriter(os.path.join(results_folder, f'results_{tag}.xlsx')) as writer:
        # to store the dataframe in specified sheet
        df_meansd.to_excel(writer, sheet_name="overview")
        # excel cannot deal with too long sheet names
        # for name, df in csv_dfs.items():
        #     df.to_excel(writer, sheet_name=name)


# compile_summary_table('/Users/linda/Documents/PhD-local/BTS/results-midl/per pid/xylabel')
# compile_summary_table('/Users/linda/Documents/PhD-local/BTS/results-midl/per pid/xylabel-0.42', tag='0.42')
# compile_summary_table('/Users/linda/Documents/PhD-local/BTS/results-midl/per pid/xylabeldino-48')
#compile_summary_table('/Users/linda/Documents/PhD-local/BTS/results-midl/per pid/xylabeldino-200')


#%%

# #%%
# results_folder = '/Users/linda/Documents/PhD-tmp/results cv10 bts'
#
# files = glob.glob(f'{results_folder}/**/metrics.csv', recursive=True)
# file_ids = [re.search(r'\d\/(.*)\/version', f).group(1) for f in files]
#
# dfs = {fid: pd.read_csv(f).filter(regex='test') for fid, f in zip(file_ids, files)}
# dfs_test = {k: v.iloc[-1] for k, v in dfs.items()}
#
# test_metrics = pd.DataFrame.from_dict(dfs_test)
# #%%
# # get the information from the experiment name
# gnns_names = [i.split(r'-')[0] for i in test_metrics.columns]
# graph_f_names = [re.search(r'(?:-(.*))-cv', i).group(1) for i in test_metrics.columns]
# edgef_names = [re.search(r'(.*)-([rdx].*)', i).group(1) for i in graph_f_names]
# feature_names = [re.search(r'(.*)-([rdx].*)', i).group(2) for i in graph_f_names]
# cv = [re.search(r'cv(\d*)', i).group(1) for i in test_metrics.columns]
# run = [re.search(r'run(\d*)', i).group(1) for i in test_metrics.columns]
#
# info_df = pd.DataFrame.from_dict({'gnn': gnns_names,
#                                   'edgef': edgef_names,
#                                   'features': feature_names,
#                                   'cv': cv,
#                                   'run': run},
#                                  orient='index', columns=test_metrics.columns)
#
# df_complete = pd.concat([info_df, test_metrics])
# # out_df.to_excel(os.path.join(results_folder, 'overview.xlsx'))
#
#
# # %%
#
# per_exp = {feat: {edgef: {gnn: {f'cv{cv}': df4.drop(['gnn', 'features', 'edgef', 'cv'], axis=1).set_index('run')
#                                 for cv, df4 in df3.groupby(['cv'])}
#                           for gnn, df3 in df2.groupby(['gnn'])}
#                   for edgef, df2 in df1.groupby(['edgef'])}
#            for feat, df1 in df_complete.T.groupby(['features'], axis=0)}
#
# # %%
# # create a multi-index df with the mean and std over the cross validation sets (average over cv of average over runs)
# feat_dfs = []
# feat_colnames = []
# edgef_colnames = []
# gnn_colnames = []
# for feat, edgef_dic in per_exp.items():
#     dfs_edgef = []
#     for edgef, gnn_dic in edgef_dic.items():
#         dfs_gnn = []
#         for gnn, cv_dic in gnn_dic.items():
#             dfs_cv = []
#             gnn_colnames.append(gnn)
#             edgef_colnames.append(edgef)
#             feat_colnames.append(feat)
#             for gnn, df_runs in cv_dic.items():
#                 sum_run = pd.concat([df_runs.mean(axis=0), df_runs.std(axis=0)], axis=1).drop(['test_loss']) * 100
#                 sum_run.columns = ['mean', 'std']
#                 dfs_cv.append(sum_run['mean'])
#             df_cv = pd.concat(dfs_cv, axis=1)
#             df_cv = pd.concat([df_cv.mean(axis=1), df_cv.std(axis=1)], axis=1)
#             df_cv.columns = ['mean', 'std']
#             df_cv = df_cv.T.unstack()
#             dfs_gnn.append(df_cv)
#
#         dfs_edgef.append(pd.concat(dfs_gnn, axis=1))
#     # multicol = pd.MultiIndex.from_product([edgef_dic.keys(), gnn_dic.keys()], names=["edge_fct", "GNN"])
#     df_feat = pd.concat(dfs_edgef, axis=1)
#     feat_dfs.append(df_feat)
#
# #%%
#
# # multicol = pd.MultiIndex.from_product([per_exp.keys(), edgef_colnames, gnn_colnames],
# #                                       names=["features", "edge_fct", "GNN"])
# tuples = list(zip(*[feat_colnames, edgef_colnames, gnn_colnames]))
# multicol = pd.MultiIndex.from_tuples(tuples, names=["features", "edge_fct", "GNN"])
# df_overview = pd.concat(feat_dfs, axis=1)
# df_overview.columns = multicol
#
# #%% create a excel writer object
# with pd.ExcelWriter(os.path.join(results_folder, 'results_summary.xlsx')) as writer:
#     # use to_excel function and specify the sheet_name and index
#     # to store the dataframe in specified sheet
#     df_overview.T.to_excel(writer, sheet_name="overview")
#     df_complete.to_excel(writer, sheet_name="full")
#
