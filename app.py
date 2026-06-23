import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import ocr_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load OCR engine on startup
    model_path = "saved_model/crnn_best.pth"
    csv_paths = [
        "DataSet/Training/training_labels.csv",
        "DataSet/Validation/validation_labels.csv",
        "DataSet/Testing/testing_labels.csv",
    ]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
    ocr_engine.load_engine(model_path=model_path, csv_paths=csv_paths)
    yield

app = FastAPI(title="PharmaLens OCR", lifespan=lifespan)

os.makedirs("templates", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join("templates", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

@app.post("/predict")
def predict(image: UploadFile = File(...)):
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG images are supported.")

    temp_path = os.path.join("temp_uploads", f"temp_{image.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        result = ocr_engine.predict(temp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
