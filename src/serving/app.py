from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import io

from torchvision import transforms

from src.model.model import Model
from src.common.config import load_config
from src.common.io import load_model_weights
import os

"""
Initialize and load trained model at API startup.

This ensures:
- Model is loaded only once (efficient inference)
- Consistent state across all requests
- Reduced latency compared to per-request loading

Raises:
    RuntimeError: If trained model checkpoint is not found.
"""
# Initialize app
app = FastAPI(title="MedMNIST Inference API")

# Load config + model ONCE at startup
config = load_config("config.yaml")

model_type = config.get("model", {}).get("type", "resnet18")

model = Model(
    num_classes=config["model"]["num_classes"],
    device=config["training"]["device"],
    model_type=model_type,
)

model_path = f"{config['output']['model_dir']}/latest.pt"

if not os.path.exists(model_path):
    raise RuntimeError("Model not found. Run training before serving.")

load_model_weights(model, model_path, device=config["training"]["device"])


# Define inference transform (NO augmentation)
# Choose inference transform based on model type
if model_type == "resnet18":
    # ResNet typically expects larger inputs (224x224)
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
else:
    # Custom CNN / MedMNIST default
    inference_transform = transforms.Compose([
        transforms.Resize((28, 28)),  # MedMNIST standard size
        transforms.ToTensor(),
    ])


@app.get("/")
def root():
    """
    Health check endpoint for the inference API.

    Returns:
        dict: Simple message confirming the API is running..
    """
    return {"message": "MedMNIST model is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Perform real-time image classification via API.

    Accepts an uploaded image file, applies preprocessing, and returns
    the predicted class along with confidence score.

    Args:
        file (UploadFile): Image file uploaded via HTTP request.

    Returns:
        dict: Prediction response containing:
            - prediction_index (int): Predicted class index
            - prediction_label (str): Human-readable class label (if available)
            - confidence (float): Probability of predicted class

    Workflow:
        1. Read and decode uploaded image
        2. Apply inference preprocessing pipeline
        3. Move tensor to model device
        4. Perform forward pass
        5. Compute softmax probabilities
        6. Extract top prediction and confidence

    Raises:
        RuntimeError: If an error occurs during inference.
    """

    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Apply transforms
    tensor = inference_transform(image).unsqueeze(0)

    # Move to device
    tensor = tensor.to(model.device)

    # Run inference
    try:
        with torch.no_grad():
            outputs = model.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
    except Exception as e:
        raise RuntimeError(f"Error during inference: {e}")
    
    # Resolve class name if provided in config
    class_names = config.get("model", {}).get("class_names")
    pred_label = None
    if isinstance(class_names, (list, tuple)) and 0 <= pred < len(class_names):
        pred_label = class_names[pred]
    else:
        pred_label = str(pred)

    return {
        "prediction_index": pred,
        "prediction_label": pred_label,
        "confidence": float(torch.max(probs).item())
    }