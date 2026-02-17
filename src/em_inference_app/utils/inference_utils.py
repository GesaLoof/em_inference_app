import torch
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_model(config, device):
    from em_nuclear_segmentation.models.unet import UNet

    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels
    )
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(
    image_path: str,
    model: torch.nn.Module,
    config,
    device: torch.device
):
    # Load image
    original_image = Image.open(image_path)
    image_np = np.array(original_image)

    # uint16 scaling
    if image_np.dtype == np.uint16:
        image_np = (image_np / 65535.0 * 255).astype(np.uint8)

    # Histogram matching
    if config.use_histogram_matching and config.histogram_reference_image:
        ref = Image.open(config.histogram_reference_image)
        ref_np = np.array(ref)

        if ref_np.dtype == np.uint16:
            ref_np = (ref_np / 65535.0 * 255).astype(np.uint8)

        image_np = match_histograms(image_np, ref_np, channel_axis=None)

    # Ensure single channel
    if image_np.ndim == 3:
        image_np = image_np[..., 0]

    # Albumentations transforms
    transform = A.Compose([
        A.Resize(config.resize_height, config.resize_width),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

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
                 > config.prediction_threshold
        ).astype(np.uint8) * 255

    # Resize back to original size
    orig_w, orig_h = original_image.size
    pred_mask_resized = Image.fromarray(pred_mask).resize(
        (orig_w, orig_h),
        resample=Image.NEAREST
    )

    return pred_mask_resized
