"""CLI entry point: python -m tabicl.train"""
from tabicl.train._train_config import build_parser
from tabicl.train._run import Trainer
from torch.multiprocessing import set_start_method

if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    trainer = Trainer(config)
    trainer.train()
