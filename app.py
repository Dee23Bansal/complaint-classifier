from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from huggingface_hub import hf_hub_download
import torch
import pandas as pd
import joblib
import os
import io
import requests

app = FastAPI()

model_path = "Dee2Bansal/sgrs-comp-classifier-fine-tuned"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
label_encoder_url = f"https://huggingface.co/{model_path}/resolve/main/label_encoder.pkl"
response = requests.get(label_encoder_url)
le = joblib.load(io.BytesIO(response.content))

labels = le.classes_

class ComplaintRequest(BaseModel):
    text: str
# 🚀 Health check
@app.get("/")
def root():
    return {"message": "Complaint Priority API is running."}

@app.post("/predict-priority")
async def predict_priority(req: ComplaintRequest):
    if not req.text.strip():
        return {"error": "Empty input"}

    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return {
        "priority": labels[pred],
        "confidence": round(probs[0][pred].item(), 3)
    }
