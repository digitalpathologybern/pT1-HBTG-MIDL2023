# Utils
import argparse
import pandas as pd
import glob
import os


def _remove_MIN_MAX(df):
    df = df[~df.index.str.contains("__MIN")]
    df = df[~df.index.str.contains("__MAX")]
    return df


def compile_results(csv_folder, dataset_names=None, output_folder=None):
    if output_folder is None:
        output_folder = csv_folder

    df_mean = pd.concat([pd.read_csv(filepath).T.drop("Step") for filepath in glob.glob(os.path.join(csv_folder, '*mean*.csv'))])
    df_sd = pd.concat([pd.read_csv(filepath).T.drop("Step") for filepath in glob.glob(os.path.join(csv_folder, '*sd*.csv'))])
    df_hparam = pd.concat([pd.read_csv(filepath).T.drop("Step") for filepath in glob.glob(os.path.join(csv_folder, '*param*.csv'))])

    # remove the MIN / MAX entries
    df_mean = _remove_MIN_MAX(df_mean)
    df_sd = _remove_MIN_MAX(df_sd)
    df_hparam = _remove_MIN_MAX(df_hparam)

    # remove the different endings in the row names and merge
    df_mean.index = [n.split(' - ')[0] for n in df_mean.index.values]
    df_sd.index = [n.split(' - ')[0] for n in df_sd.index.values]
    df_hparam.index = [n.split(' - ')[0] for n in df_hparam.index.values]
    dfs_all = df_mean.merge(df_sd, left_index=True, right_index=True, how='inner')
    dfs_all = dfs_all.merge(df_hparam, left_index=True, right_index=True, how='inner')
    dfs_all.columns = ['mean', 'sd', 'hparam']

    # SingleGraph-gxl-D2-graphsage_singlegraph-20
    if dataset_names is not None:
        dfs_per_ds = {f'{ds}-all': dfs_all[dfs_all.index.str.contains(f'{ds}-')].sort_index() for ds in dataset_names}
        # insert column for nb neurons and experiment-id
        _ = {ds: df.insert(0, 'nb_layers', [int(row_name.split('-')[-1]) for row_name in df.index.values]) for ds, df in dfs_per_ds.items()}
        _ = {ds: df.insert(0, 'dataset', [ds]*len(df.index.values)) for ds, df in dfs_per_ds.items()}
        _ = {ds: df.insert(0, 'architecture', [row_name.split('-')[-2] for row_name in df.index.values]) for ds, df in dfs_per_ds.items()}
        _ = {ds: df.insert(0, 'experiment_name', [row_name.split('-')[0] for row_name in df.index.values]) for ds, df in dfs_per_ds.items()}

        # rename column names
        dfs_per_ds = {ds: df.rename({row_name: f'{df.loc[row_name,"experiment_name"]}-{df.loc[row_name,"architecture"]}'
                                     for row_name in df.index.values}, axis='index') for ds, df in dfs_per_ds.items()}
        # sort the columns first by column name and then by nb neurons
        dfs_per_ds = {ds: df.sort_values(['experiment_name', 'architecture', 'nb_layers'], ascending=[True, True, True]) for ds, df in dfs_per_ds.items()}

        # make an overview with just the mean (easier to plot)
        dfs_per_ds_overview = {ds: df.drop(['sd', 'experiment_name', 'architecture', 'dataset'], axis=1) for ds, df in dfs_per_ds.items()}
        df = dfs_per_ds_overview['gxl-D10-all']

        # write to file
        writer = pd.ExcelWriter(os.path.join(output_folder, 'results.xlsx'), engine='xlsxwriter')
        for ds, df in dfs_per_ds.items():
            df.to_excel(writer, sheet_name=ds)
        pd.concat([dfs_per_ds[k] for k in sorted(dfs_per_ds.keys())]).to_excel(writer, sheet_name='all')
        writer.save()

    else:
        # write to file
        dfs_all.insert(0, 'nb_layers', [int(row_name.split('-')[-1]) for row_name in dfs_all.index.values])
        dfs_all.insert(0, 'architecture', [row_name.split('-')[-2] for row_name in dfs_all.index.values])
        dfs_all.insert(0, 'experiment_name', [row_name.split('-')[0] for row_name in dfs_all.index.values])
        dfs_all.sort_values(['experiment_name', 'architecture', 'nb_layers'], ascending=[True, True, True])

        dfs_all = dfs_all.sort_values(['experiment_name', 'architecture', 'nb_layers'], ascending=[True, True, True])
        writer = pd.ExcelWriter(os.path.join(output_folder, 'results.xlsx'), engine='xlsxwriter')
        dfs_all.to_excel(writer)
        writer.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile the CSV results from wandb into a nice exel and figures.')
    parser.add_argument('--csv-folder', type=str, help='Path to the folder with the CSV files to be parsed')
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path where outputs should be save to (default is same as csv-folder).')
    args = parser.parse_args()

    dataset_names = None
    #dataset_names = ['gxl-D10', 'gxl-D25', 'gxl-D50']
    #dataset_names = ['gxl-D50']
    compile_results(csv_folder=args.csv_folder, output_folder=args.csv_folder, dataset_names=dataset_names)