from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model_def import LSTM_reg

# -------------------- FASTAPI APP --------------------
app = FastAPI(title="Stock Prediction with News Sentiment API")

# -------------------- MODEL CONFIG (MUST MATCH TRAINING) --------------------
input_size = 4
hidden_size = 60
num_layers = 1
num_classes = 1
dropout = 0.0
fc_size = 310

# -------------------- LOAD MODEL --------------------
model = LSTM_reg(
    num_classes=num_classes,
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    fc_size=fc_size
)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# -------------------- INPUT SCHEMA --------------------
class StockInput(BaseModel):
    f1: float
    f2: float
    f3: float
    f4: float

# -------------------- ROUTES --------------------
@app.get("/")
def root():
    return {"status": "API running successfully"}

@app.post("/predict")
def predict(data: StockInput):
    # Convert input to tensor
    x = torch.tensor(
        [[data.f1, data.f2, data.f3, data.f4]],
        dtype=torch.float32
    )

    # Reshape for LSTM -> (batch, seq_len, features)
    x = x.unsqueeze(1)

    # Prediction
    with torch.no_grad():
        output = model(x)

    return {
        "prediction": output.item()
    }
