import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
from pathlib import Path
import yaml

app = FastAPI(title="MLOps PyTorch Serving API")

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config():
    config_path = Path("configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path("/app/configs/training_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

@app.on_event("startup")
def startup_event():
    global model
    try:
        config = load_config()
        from model import get_model
        
        model = get_model(
            architecture=config["model"]["architecture"],
            num_classes=config["model"]["num_classes"]
        ).to(device)
        
        checkpoint_dir = Path(config["output"]["checkpoint_dir"])
        model_path = checkpoint_dir / config["output"]["model_name"]
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: Checkpoint not found at {model_path}. Model running with uninitialized weights.")
            
        model.eval()
    except Exception as e:
        print(f"Error loading model during startup: {e}")

@app.get("/health")
def health_check():
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    raise HTTPException(status_code=503, detail="Model not loaded")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        results = {classes[i]: round(float(probabilities[i]), 4) for i in range(10)}
        
        return {"filename": file.filename, "probabilities": results}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))