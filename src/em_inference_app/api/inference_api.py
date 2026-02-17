# app/main.py

from fastapi import FastAPI, UploadFile, File, Form
from em_inference_app.utils.inference_utils import predict
from app.schemas import PredictParams
import tempfile
import shutil

app = FastAPI()

@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(...),
    threshold: float = Form(0.5),
    device: str = Form("cpu")
):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        shutil.copyfileobj(image.file, tmp)
        temp_path = tmp.name

    # Call your core inference function
    result = predict(
        temp_path,
        threshold=threshold,
        device=device
    )

    return {"result": result}
