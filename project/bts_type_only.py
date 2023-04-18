import torch_geometric
from torch_geometric.transforms import RandomShear, RandomScale, RandomRotate


from util.arg_parsers import GraphClassificationGxlCLArguments
from util.custom_transforms import NodeDrop
from project.bts_experiments import Bts
from project import project_util


class BtsTypeOnly(Bts):
    def __init__(self, multi_run, **kwargs):
        super(BtsTypeOnly, self).__init__(multi_run=multi_run, **kwargs)
        self.set_metrics()

    def get_train_transforms(self):
        return torch_geometric.transforms.Compose([
            NodeDrop(p=0.1)
        ])

    def get_val_transforms(self):
        return self.get_train_transforms()

    def get_test_transforms(self):
        return None


if __name__ == '__main__':
    # ------------
    # args, setup and logging
    # ------------
    kwargs = project_util.get_kwargs(GraphClassificationGxlCLArguments())

    BtsTypeOnly(**kwargs).main()
