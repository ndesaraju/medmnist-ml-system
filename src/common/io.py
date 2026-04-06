import os
import json
from datetime import datetime
import torch


def get_timestamp() -> str:
    """
    Generate a timestamp string for versioning.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    """
    Create directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def save_model(model, config, filename: str = None) -> str:
    """
    Save model weights.

    Returns:
        str: path to saved model
    """
    model_dir = config["output"]["model_dir"]
    ensure_dir(model_dir)

    if filename is None:
        filename = f"model_{get_timestamp()}.pt"

    path = os.path.join(model_dir, filename)

    torch.save(model.model.state_dict(), path)

    # Optional: update latest pointer
    latest_path = os.path.join(model_dir, "latest.pt")
    torch.save(model.model.state_dict(), latest_path)

    print(f"Model saved to: {path}")

    return path


def load_model_weights(model, path: str, device="cpu"):
    """
    Load weights into model instance.
    """
    state_dict = torch.load(path, map_location=device)
    model.model.load_state_dict(state_dict)
    model.model.eval()


def save_metrics(metrics: dict, config, filename: str = None) -> str:
    """
    Save metrics to JSON.

    Returns:
        str: path to saved metrics
    """
    metrics_dir = config["output"]["metrics_dir"]
    ensure_dir(metrics_dir)

    if filename is None:
        filename = f"metrics_{get_timestamp()}.json"

    path = os.path.join(metrics_dir, filename)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to: {path}")

    return path


def load_metrics(path: str) -> dict:
    """
    Load metrics from JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)