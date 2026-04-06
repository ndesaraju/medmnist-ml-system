import yaml
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path (str): Path to config.yaml

    Returns:
        dict: configuration dictionary
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_device(config: Dict[str, Any]):
    """
    Utility to fetch device from config with fallback.

    Args:
        config (dict)

    Returns:
        torch.device
    """
    import torch

    device_str = config.get("training", {}).get("device", "cpu")
    return torch.device(device_str)
