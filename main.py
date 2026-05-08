from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

import torch
import io
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms, models

import torch.nn as nn

# ======================
# 1. SETTINGS
# ======================
MODEL_PATH = "best_model.pth"
IMG_SIZE = 224

CLASSES = ["WithMask", "WithoutMask"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 2. FACE DETECTOR
# ======================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ======================
# 3. LOAD MODEL
# ======================
def load_model(path):

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(model.last_channel, 2)

    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model(MODEL_PATH)

# ======================
# 4. PREPROCESSING
# ======================
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ======================
# 5. APP
# ======================
app = FastAPI(title="Face Mask Detection API")

# ======================
# 6. HOME PAGE
# ======================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Face Mask Detection</title>

    <style>
        body {
            font-family: Arial;
            text-align: center;
            background-color: #f4f4f4;
            margin-top: 20px;
        }

        video {
            width: 400px;
            height: 400px;
            border: 3px solid black;
            border-radius: 10px;
            object-fit: cover;
            margin-top: 15px;

            /* ✅ MIRROR CAMERA */
            transform: scaleX(-1);
        }

        button {
            margin-top: 15px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
            background-color: #007bff;
            color: white;
        }

        button:hover {
            background-color: #0056b3;
        }

        #result {
            margin-top: 20px;
            font-size: 24px;
            font-weight: bold;
        }

        #note-box {
            position: fixed;
            top: 15px;
            right: 15px;
            width: 300px;
            background-color: #fff3cd;
            border: 2px solid #ffcc00;
            padding: 15px;
            border-radius: 10px;
            text-align: left;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.2);
            font-size: 14px;
        }
    </style>
</head>

<body>

<div id="note-box">
    <b>📌 Image Guidelines</b>
    <ul>
        <li>Only one face should appear</li>
        <li>Keep face close to camera</li>
        <li>Good lighting</li>
        <li>Avoid blur</li>
    </ul>
</div>

<h1>Face Mask Detection</h1>

<h2>📷 Camera</h2>

<video id="video" autoplay></video>

<br>

<button onclick="captureImage()">Capture & Predict</button>

<canvas id="canvas" width="400" height="400" style="display:none;"></canvas>

<h2>🖼 Upload Image</h2>

<input type="file" id="uploadInput">
<br>
<button onclick="uploadImage()">Upload & Predict</button>

<div id="result"></div>

<script>

// ======================
// CAMERA
// ======================
const video = document.getElementById("video");

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
});

// ======================
// CAPTURE
// ======================
async function captureImage() {

    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");

    // ✅ FIX MIRROR CAPTURE (important!)
    context.translate(canvas.width, 0);
    context.scale(-1, 1);

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async function(blob) {

        let formData = new FormData();
        formData.append("file", blob, "camera.jpg");

        let response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        let result = await response.json();

        document.getElementById("result").innerHTML =
            "Prediction: " + result.class +
            "<br><br>Confidence: " + result.confidence.toFixed(4);

    }, "image/jpeg");
}

// ======================
// UPLOAD
// ======================
async function uploadImage() {

    const input = document.getElementById("uploadInput");

    if (input.files.length === 0) {
        alert("Please choose an image first.");
        return;
    }

    let formData = new FormData();
    formData.append("file", input.files[0]);

    let response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    let result = await response.json();

    document.getElementById("result").innerHTML =
        "Prediction: " + result.class +
        "<br><br>Confidence: " + result.confidence.toFixed(4);
}

</script>

</body>
</html>
"""

# ======================
# 7. PREDICTION
# ======================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_np = np.array(img)

        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # ======================
        # BETTER FACE DETECTION
        # ======================
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(60, 60)
        )

        # ======================
        # FALLBACK: flipped image detection
        # ======================
        if len(faces) == 0:
            gray_flipped = cv2.flip(gray, 1)

            faces = face_cascade.detectMultiScale(
                gray_flipped,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=(60, 60)
            )

        # ======================
        # CROP FACE
        # ======================
        if len(faces) > 0:

            x, y, w, h = faces[0]

            padding = int(0.35 * w)

            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)

            x2 = min(x + w + padding, img_cv.shape[1])
            y2 = min(y + h + padding, img_cv.shape[0])

            face_crop = img_np[y1:y2, x1:x2]

            img = Image.fromarray(face_crop)

        # ======================
        # PREPROCESS
        # ======================
        tensor = preprocess(img).unsqueeze(0).to(device)

        # ======================
        # PREDICT
        # ======================
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        predicted_class = CLASSES[probs.argmax().item()]
        confidence = float(probs.max().item())

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                CLASSES[i]: float(probs[i].item())
                for i in range(len(CLASSES))
            }
        }

    except Exception as e:
        return {"error": str(e)}

# ====================== # RUN COMMAND # ====================== 
# for reload: 
# uvicorn main:app --reload --port 8000 

# OPEN:
# http://127.0.0.1:8000