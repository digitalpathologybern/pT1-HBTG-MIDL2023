import json
import pytorch_lightning as pl


def get_kwargs(parser):
    parser = parser.get_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    kwargs = parser.parse_args().__dict__

    if kwargs['config_json'] is not None:
        with open(kwargs['config_json']) as f:
            config = json.load(f)
        kwargs.update(config)

    return kwargs
