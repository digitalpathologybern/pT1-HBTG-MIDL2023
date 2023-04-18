# pT1-HBTG-MIDL2023
Code for publication [_Tumor Budding T-cell Graphs: Assessing the Need for Resection in pT1 Colorectal Cancer Patients_](https://openreview.net/forum?id=ruaXPgZCk6i) 
presented at MIDL 2023.

Bibtex:
```
@inproceedings{studer2023tumor,
  title={Tumor Budding T-cell Graphs: Assessing the Need for Resection in pT1 Colorectal Cancer Patients},
  author={Studer, Linda and Bokhorst, John-Melle and Nagtegaal, Iris and Zlobec, Inti and Dawson, Heather and Fischer, Andreas},
  booktitle={Medical Imaging with Deep Learning},
  year={2023}
}
```
## The pT1-HBTG Dataset

COMING SOON
 
## How to run: Graph Neural Network Framework using Pytorch Geometric and Pytorch-Lightning     

This framework allows you to efficiently set up experiments for the graph-level classification. It is based on [PyTorch](https://pytorch.org/get-started/locally/), 
and uses the [PyTorch Geometric library](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
for building the graph datasets and GNN architectures, and [Pytorch-Lightning](https://www.pytorchlightning.ai/)
to organize the code. [Weights & Biases](https://wandb.ai) is used for experiment logging.
   
### Installation
First, you need to set up your conda environment with the right python packages (for the full list of package versions 
see `gnn-env.yml`)
```
# create conda env
conda create --name gnn python=3.8
conda activate gnn

# install pytorch 1.11.0 with correct cuda version (check website), e.g.
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
# for cpu only for Mac
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch

# install pytorch geometric for cuda 11.2 (for other versions, check the website)
conda install pyg -c pyg

# install pytorch lightning
pip install pytorch-lightning==1.6.3

# install other packages
conda install -c conda-forge wandb==0.12.17
conda install seaborn==0.11.2

# torchmetrics should already be installed, if not run
conda install -c conda-forge torchmetrics
```

### Dataset parsing
- **GXL format data**: [gxl](https://en.wikipedia.org/wiki/GXL?oldformat=true) is a type of xml format developed for 
  graphs. The class and dataset splits can be set up in two different ways:
    - Using a `train.cxl`, `valid.cxl` and `test.cxl` file (as for the IAMDB Graph datasets)
    - Using a folder structure with `{train/test/val}/{class1, class2, ...}/*.gxl`.
    - Using a `json` file that provides a cross-validation fold split, with the class labels.

### Set up your experiment
To set up your experiments you have to create your own experiment file in the `project` folder and set up a 
`class` that inherits from `Experiment` in `/project/experiment_template.py`. In there you specify your experimental, e.g. 
the runner you want to use, the transformations, the performance metrics, etc.

### Run your experiment
Example:
`--input-folder "/HOME/studerl/datasets/TUDataset/" --model graphsage_singlegraph --output-folder ../output-debug --experiment-name test-TU-mr --epochs 2 --batch-size 2 --gpu-id 3 --subdataset-name "COLORS-3" --multi-run 2 --wandb-project test --use-node-attr`

For the full list of command line arguments (CLA) see `util/arg_parser.py`. Arguments can be either specified via the command line,
a config `json` file, or both (parameters set in the config file overwrite the ones specified as a CLA).


### Framework structure
- `data_modeules`: Contains the PyTorch Geometric (custom) dataset loaders
- `model_modules`: Contains the graph neural network architectures
- `project`: Contains the experimental set-up for specific experiments
- `runners`: Contains the `pytorch_lightning.LightningModule` classes for a specific task (e.g. classification)
- `util`: contains utility functions, such as project specific argument parsers
- `util_scripts`: contains additional utility scripts that are separate from the experimental framework