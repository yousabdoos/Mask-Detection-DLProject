from fastapi import FastAPI, UploadFile, File
import torch
import io
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
# 2. LOAD MODEL
# ======================
def load_model(path):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH)

# ======================
# 3. PREPROCESSING
# ======================
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ======================
# 4. FASTAPI APP
# ======================
app = FastAPI(title="Face Mask Detection API")

@app.get("/")
def home():
    return {"message": "API is working"}

# ======================
# 5. PREDICTION ENDPOINT
# ======================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        tensor = preprocess(img).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        predicted_class = CLASSES[probs.argmax().item()]
        confidence = float(probs.max().item())

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                CLASSES[i]: float(probs[i].item()) for i in range(len(CLASSES))
            }
        }

    except Exception as e:
        return {"error": str(e)}
