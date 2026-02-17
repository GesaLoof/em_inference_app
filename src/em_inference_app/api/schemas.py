from pydantic import BaseModel, Field
from typing import Literal, Optional


class PredictParams(BaseModel):
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for segmentation mask"
    )

    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device to run inference on"
    )


class PredictResponse(BaseModel):
    message: str
    mask_shape: tuple[int, int]


class InferenceConfig(BaseModel):
    model_path: str
    in_channels: int = 1
    out_channels: int = 1
    resize_height: int
    resize_width: int
    prediction_threshold: float = Field(0.5, ge=0.0, le=1.0)

    use_histogram_matching: bool = False
    histogram_reference_image: Optional[str] = None

    save_visual_overlay: bool = False
    prediction_output_dir: str
