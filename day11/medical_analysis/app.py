from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()
model = tf.keras.models.load_model('model/cnn_model.h5')
class_names = ['NORMAL', 'PNEUMONIA']
image_size = (224, 224)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        pil_img = Image.open(BytesIO(contents)).convert('RGB')
        pil_img = pil_img.resize(image_size)
        img_arr = image.img_to_array(pil_img)
        x = np.expand_dims(img_arr, axis=0).astype("float32") / 255.0

        # Predict
        pred = model.predict(x, verbose=0)[0]
        prob_pneumonia = float(pred.squeeze())
        prob_normal = 1.0 - prob_pneumonia
        pred_label = class_names[int(prob_pneumonia >= 0.5)]

        return JSONResponse({
            "prediction": pred_label,
            "probabilities": {
                "NORMAL": prob_normal,
                "PNEUMONIA": prob_pneumonia
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)