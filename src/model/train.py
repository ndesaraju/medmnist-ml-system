import torch
import torch.nn as nn
import torch.optim as optim

from src.common.config import load_config
from src.preprocess.build_features import get_dataloaders
from src.model.model import Model
from src.evaluate.evaluate import evaluate
from src.evaluate.formatting import print_metrics
from src.common.io import get_timestamp, save_model, save_metrics
from src.common.utils import set_seed
import os
import json
import time


def train():
    print("Loading configuration.")
    config = load_config("config.yaml")

    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    print("Getting data loaders.")
    train_loader, val_loader, _ = get_dataloaders(config)

    print("Initializing model.")
    model = Model(
        num_classes=config["model"]["num_classes"],
        device=config["training"]["device"],
        model_type=config.get("model", {}).get("type", "resnet18"),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.parameters(), lr=config["training"]["lr"])

    print(f"Starting training with model {model.model_type}.")
    total_start = time.perf_counter()
    epoch_times = []
    for epoch in range(config["training"]["epochs"]):
        epoch_start = time.perf_counter()
        loss = model.train(train_loader, optimizer, criterion)
        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_times.append(epoch_elapsed)

        # simple ETA: remaining_epochs * last_epoch_time
        remaining = config["training"]["epochs"] - (epoch + 1)
        eta_sec = remaining * epoch_elapsed
        eta_str = f"{eta_sec:.1f}s" if remaining > 0 else "0s"

        print(f"Epoch {epoch+1}: Loss = {loss:.4f} — time: {epoch_elapsed:.2f}s; ETA: {eta_str}")

    total_elapsed = time.perf_counter() - total_start
    print(f"Total training time: {total_elapsed:.2f}s")

    preds, labels = model.predict(val_loader)
    probs, _ = model.predict_proba(val_loader)

    metrics = evaluate(labels, preds, probs)

    # Save model
    print("Saving model.")
    bs = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    epochs_cfg = config["training"]["epochs"]
    try:
        model_type = getattr(model.model, "__class__").__name__
    except Exception:
        model_type = "model"

    model_type_safe = str(model_type).lower().replace(" ", "_")
    lr_safe = str(lr).replace(".", "p")

    run_id = f"bs{bs}_model{model_type_safe}_lr{lr_safe}_ep{epochs_cfg}_{get_timestamp()}"
    model_filename = f"model_{run_id}.pt"

    # save_model will create the directory and also write a 'latest.pt' copy
    save_model(model, config, filename=model_filename)

    # Save metrics using the same run identifier to tie artifacts together
    print("Saving metrics.")
    metrics_filename = f"metrics_{run_id}.json"
    save_metrics(metrics, config, filename=metrics_filename)
    class_names = config["model"]["class_names"]
    print_metrics(metrics, class_names=class_names)
    print("Training complete.\n")


if __name__ == "__main__":
    train()