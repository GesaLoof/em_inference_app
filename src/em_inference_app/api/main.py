from fastapi import FastAPI, UploadFile, File, Form
from em_inference_app.utils.inference_utils import (
    build_model,
    build_transform,
    predict
)
from em_inference_app.utils.inference_config import InferenceConfig
import torch
import tempfile
import shutil


app = FastAPI()

# ---- SERVER STARTUP CONFIG ----
config = InferenceConfig(
    model_path="models/unet.pth",
    in_channels=1,
    out_channels=1,
    resize_height=512,
    resize_width=512,
    histogram_reference_image=None,
)

device = torch.device("cpu")  # can improve this later

model = build_model(config, device)
transform = build_transform(config)


@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(...),
    threshold: float = Form(0.5)
):
    # Convert device string
    request_device = torch.device(device_str)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        shutil.copyfileobj(image.file, tmp)
        temp_path = tmp.name

    # Run inference
    mask = predict(
        image_path=temp_path,
        model=model,
        transform=transform,
        config=config,
        device=request_device,
        prediction_threshold=threshold,
        use_histogram_matching=False,
    )

    return {"message": "Inference complete"}
