import argparse
import os

import torch
from omegaconf import OmegaConf

from src.datasets import EllipticDataset
from src.models import GAT
from src.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/elliptic.yaml",
        required=True,
        help="Config for Training",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        default=False,
        action="store_true",
        required=False,
        help="Visualizer",
    )

    args = parser.parse_args()
    config_path = args.config
    print("Loading Config...")
    config = OmegaConf.load(config_path)
    print("Config Loaded")
    trainer = Trainer(config)

    if not args.visualize:
        trainer.train()
        trainer.save(config.name)
    else:
        # Visualize
        time_step = config.visualize.step
        weights_path = config.visualize.weights_path

        print("\nVisualizing on Elliptic Dataset\n")

        dataset = EllipticDataset(config.dataset)
        config.model.input_dim = dataset.pyg_dataset().num_node_features

        model = GAT(config.model)
        if weights_path is None:
            weights_path = f"weights/{config.name}.pt"
        model.load_state_dict(torch.load(weights_path))
        trainer.model = model.double().to(config.train.device)

        trainer.visualize(
            dataset,
            time_step=time_step,
            save_to=f"results/{config.name}/{time_step}.png",
        )
