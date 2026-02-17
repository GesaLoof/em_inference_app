# app/settings.py (or similar)
from pydantic_settings import BaseSettings
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    # Paths
    MODEL_PATH: Path = PROJECT_ROOT / "models" / "segmentation_v1.pth"
    MODEL_DEVICE: str = "cuda"
    
    # API settings
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_FORMATS: list[str] = ["jpg", "png"]
    
    # Processing
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    MAX_BATCH_SIZE: int = 4
    model_config_path: str = "configs/unet.yaml"
    
    class Config:
        env_file = ".env"  # Can override via environment