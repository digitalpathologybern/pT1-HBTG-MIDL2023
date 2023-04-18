import pandas as pd
import os

#%%
# command to make std smaller font: \std{25.9}{18.3}
# \newcommand{\std}[2] {
#   #1\begin{tiny}$\,\pm\,$#2\end{tiny}
# }

file = '/Users/linda/Nextcloud/PhD/Papers and Conferences/2023-07 MIDL/results-midl/per pid/aggr10-dinotypecoord/results_aggr10-dinotypecoord.xlsx'

df_full = pd.read_excel(file, index_col=[0, 1], header=[0, 1])
df = df_full[df_full.columns.drop(list(df_full.filter(regex='Accuracy')))]
df = df[list(df.filter(regex='_c'))]

#%%

rename_gnn = {'gatv2': r'GATv2', 'gatv2_jk': 'GATv2-JK', 'gin': 'GIN', 'gin_jk': 'GIN-JK', 'graphsage': 'GraphSAGE',
              'graphsage_jk': 'GraphSAGE-JK'}
# rename_gnn = {k: fr'\multirow{{2}}{{*}}{v}' for k, v in rename_gnn.items()}


def save_latex(df, gtype, file):
    # set index
    iterables = [[rename_gnn[name.split('-', 1)[0]] for name in df.index],
                 ['Yes' if 'addf' in name else 'No' for name in df.index]]
    iterables = [(iterables[0][i], iterables[1][i]) for i in range(len(df.index))]
    multi_index = pd.MultiIndex.from_tuples(iterables, names=["GNN", "Clin. Info"])
    df.index = multi_index

    # combine mean and std
    df = df.round(1)
    metrics = sorted(list(set(df.columns.get_level_values(0))))
    d_formatted = {m: df[m].apply(lambda row: fr'\std{{{row.loc["mean"]}}}{{{row.loc["std"]}}}', axis=1) for m in metrics}
    d_formatted = pd.DataFrame.from_dict(d_formatted)

    # save
    d_latex = d_formatted.to_latex(escape=False, multirow=True)
    d_latex.replace('\cline{1-8}', '')
    with open(file.replace('.xlsx', f'-{gtype}-latex.txt'), 'w') as f:
        f.write(d_latex)


for gtype in list(set(df.index.get_level_values(0))):
    save_latex(df.loc[gtype], gtype, file)
