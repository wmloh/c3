import os
import argparse
import shutil
import json

from typing import Any

OPENML_CHOICES = ["C3", "linucb", "neuralucb", "lts", "neuralts", "squarecb", "neurallinear"]
MIND_CHOICES = ["C3", "bayeslr","twotower", "swlinucb", "dlinucb", "gpforget", "onlinenucb", "onlinents", "crb"]


def parse_eval_openml(save_args: bool = True) -> tuple[argparse.Namespace, dict[str, Any]]:
    """
    Parse command line arguments and file configs for eval_openml.

    Args:
        save_args: If True, save args and config into the experiment directory

    Returns:
        Namespace object containing passed arguments, and config dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="Directory to store results", type=str)
    parser.add_argument("algorithm", help="Choice of algorithm", type=str, choices=OPENML_CHOICES)
    parser.add_argument("config", help="JSON file containing configurations file", type=str)
    parser.add_argument("--seed", help="Seed number", default=42, type=int)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, str(args.seed)), exist_ok=True)

    if save_args:
        # copies file directly
        shutil.copy(args.config, os.path.join(args.exp_dir, str(args.seed), "config.json"))

        # save command line args as a JSON
        with open(os.path.join(args.exp_dir, str(args.seed), "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    return args, config


def parse_eval_mind(save_args: bool = True) -> tuple[argparse.Namespace, dict[str, Any], dict[str, Any]]:
    """
    Parse command line arguments and file configs for eval_mind.

    Args:
        save_args: If True, save args and config into the experiment directory

    Returns:
        Namespace object containing passed arguments, config dict, and MIND model args dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="Directory to store results", type=str)
    parser.add_argument("algorithm", help="Choice of algorithm", choices=MIND_CHOICES, type=str)
    parser.add_argument("config", help="JSON file containing experiment configurations file", type=str)
    parser.add_argument("model_config", help="JSON file containing model configurations file", type=str)
    parser.add_argument("train_data_dir", help="Directory of training MIND dataset", type=str)
    parser.add_argument("val_data_dir", help="Directory of validation MIND dataset", type=str)
    parser.add_argument("embedding_dir", help="Directory containing embedding dictionary", type=str)
    parser.add_argument("--seed", help="Seed number", default=42, type=int)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    with open(args.model_config, "r") as f:
        params = json.load(f)

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, str(args.seed)), exist_ok=True)

    if save_args:
        # copies file directly
        shutil.copy(args.config, os.path.join(args.exp_dir, str(args.seed), "config.json"))
        shutil.copy(args.model_config, os.path.join(args.exp_dir, str(args.seed), "modelargs.json"))

        # save command line args as a JSON
        with open(os.path.join(args.exp_dir, str(args.seed), "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    return args, config, params
