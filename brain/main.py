from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import tf_keras
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "alzheimer_cnn_efficientnetb0.h5"
CLASSES = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
IMG_SIZE = (224, 224)

print("Loading model...")
model = tf_keras.models.load_model(MODEL_PATH)
print("Model loaded.")


@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]
        probs = {CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASSES))}
        top_class = CLASSES[int(np.argmax(preds))]
        top_conf = round(float(np.max(preds)) * 100, 2)

        return JSONResponse({
            "prediction": top_class,
            "confidence": top_conf,
            "probabilities": probs
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
