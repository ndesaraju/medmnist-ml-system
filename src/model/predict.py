from src.model.model import Model
from src.common.config import load_config
from src.preprocess.build_features import get_dataloaders
from src.evaluate.evaluate import evaluate
from src.evaluate.formatting import print_metrics

import os


def predict(model_path=None, run_evaluation=False):
    """
    Run inference on the test dataset using a trained model.

    Loads the model from disk (default: latest checkpoint), performs
    predictions on the test set, using the same preprocessing pipeline as training,
    and optionally computes evaluation metrics.

    Args:
        model_path (str, optional): Path to the model checkpoint.
            If None, defaults to "latest.pt" in the configured model directory.
        run_evaluation (bool, optional): Whether to compute and print
            evaluation metrics. Defaults to False.

    Returns:
        List[int]: Predicted class indices for the test dataset.

    Workflow:
        1. Load configuration
        2. Initialize test DataLoader
        3. Load trained model
        4. Run inference on test set
        5. (Optional) Compute evaluation metrics
    """
    config = load_config("config.yaml")

    _, _, test_loader = get_dataloaders(config)

    # Load model
    model = Model(
        num_classes=config["model"]["num_classes"],
        device=config["training"]["device"],
        model_type=config["model"]["type"],
    )

    if model_path is None:
        model_path = os.path.join(config["output"]["model_dir"], "latest.pt")

    model.load(model_path)
    print(f"Loaded model from: {model_path}")

    preds, labels = model.predict(test_loader)
    print(f"Number of predictions: {len(preds)}")

    if run_evaluation:
        probs, _ = model.predict_proba(test_loader)
        metrics = evaluate(labels, preds, probs)

        # get class names from config for nicer formatting
        class_names = config["model"]["class_names"]
        print_metrics(metrics, class_names=class_names)

    print("Prediction complete.\n")

    return preds


if __name__ == "__main__":
    predict(run_evaluation=True)