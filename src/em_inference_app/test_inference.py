# test_inference.py

import torch
from app.config import load_config
from em_nuclear_segmentation.utils.inference_utils import build_model, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = load_config("config.yaml")
model = build_model(config, device)

result = predict(
    image_path="test_image.tif",
    model=model,
    config=config,
    device=device
)

print(result.size)
result.show()
