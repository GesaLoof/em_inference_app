import torch
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from em_inference_app.utils.inference_config import InferenceConfig
from em_inference_app.models.unet import UNet
from pathlib import Path


def build_model(config: InferenceConfig, device: torch.device):
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels
    )

    BASE_DIR = Path(__file__).resolve().parents[3]
    model_path = BASE_DIR / config.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    return model


def build_transform(config: InferenceConfig):
    return A.Compose([
        A.Resize(config.resize_height, config.resize_width),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

def predict(
    image_path: str,
    model: torch.nn.Module,
    transform,
    config: InferenceConfig,
    device: torch.device,
    prediction_threshold: float,
    use_histogram_matching: bool,
) -> np.ndarray:

    # Load image
    original_image = Image.open(image_path)
    image_np = np.array(original_image)

    # uint16 scaling
    if image_np.dtype == np.uint16:
        image_np = (image_np / 65535.0 * 255).astype(np.uint8)

    # Histogram matching (optional per request)
    if use_histogram_matching and config.histogram_reference_image:
        ref = Image.open(config.histogram_reference_image)
        ref_np = np.array(ref)

        if ref_np.dtype == np.uint16:
            ref_np = (ref_np / 65535.0 * 255).astype(np.uint8)

        image_np = match_histograms(image_np, ref_np, channel_axis=None)

    # Ensure single channel
    if image_np.ndim == 3:
        image_np = image_np[..., 0]

    # Apply transform
    sample = transform(image=image_np)
    tensor = sample["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output)

        pred_mask = (
            probs.squeeze(0)
                 .squeeze(0)
                 .cpu()
                 .numpy()
                 > prediction_threshold
        ).astype(np.uint8) * 255

    # Resize back to original size
    orig_w, orig_h = original_image.size
    pred_mask_resized = Image.fromarray(pred_mask).resize(
        (orig_w, orig_h),
        resample=Image.NEAREST
    )

    return np.array(pred_mask_resized)