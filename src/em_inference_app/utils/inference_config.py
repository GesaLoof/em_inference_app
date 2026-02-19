from dataclasses import dataclass
from typing import Optional

@dataclass
class InferenceConfig:
    model_path: str
    in_channels: int = 1
    out_channels: int = 1
    resize_height: int = 512
    resize_width: int = 512
    histogram_reference_image: Optional[str] = None
