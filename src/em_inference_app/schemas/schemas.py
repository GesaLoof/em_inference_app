from pydantic import BaseModel, Field
from typing import Literal


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
    use_histogram_matching: bool = False
    save_visual_overlay: bool = False 


class PredictResponse(BaseModel):
    message: str
    mask_shape: tuple[int, int]
    mask_path: str
    overlay_path: str | None
