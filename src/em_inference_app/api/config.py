import yaml
from app.schemas import InferenceConfig


def load_config(path: str) -> InferenceConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return InferenceConfig(**raw)