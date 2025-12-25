from fastapi import FastAPI, Request, Security, HTTPException
from fastapi.responses import JSONResponse, Response
import joblib
import numpy as np
from pydantic import BaseModel, validator, ValidationError
import os, time, logging, json
from datetime import datetime
from dotenv import load_dotenv
import os
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
import torch
import torch.nn as nn
import sys

#sys.setrecursionlimit(5000)

app = FastAPI(title='Insurance Charges Prediction')

rf_model = None
xgb_model = None

load_dotenv()
API_KEY = os.getenv("API_KEY")

api_key_header = APIKeyHeader(name=API_KEY, auto_error=False)

#   print(os.getcwd())

@app.on_event("startup")
def load_models():
    global rf_model, xgb_model

    rf_model = joblib.load("./models/rf_regressor.pkl")
    xgb_model = joblib.load("./models/xgb_regressor.pkl")

    print("Models loaded successfully")

'''
checkpoint = torch.load("./models/regression_model.pt", map_location="cpu")

INPUT_SIZE = checkpoint["input_size"]

print(checkpoint)
class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_size, n_layers, dropout_rate):
        super().__init__()
        layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.ReLU()
            ])
            in_dim = hidden_size

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = RegressionNet(
    input_dim=INPUT_SIZE,
    hidden_size=256,      
    n_layers=2,         
    dropout_rate=0.3360      
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(model)
'''

class InsuranceData(BaseModel):
    age:        int
    bmi:        float
    bmi_category_normal: float
    bmi_category_obese: float
    bmi_category_overweight: float
    bmi_category_underweight: float
    children:   int
    sex_0 :     float
    sex_1 :     float
    region_0:   float 
    region_1:   float
    region_2:   float
    region_3:   float
    smoker_0:   float
    smoker_1:   float
    age_group_0_25: float
    age_group_26_35: float
    age_group_36_45: float
    age_group_46_55: float
    age_group_55_65: float

    @validator("age")
    def age_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("age must be greater than 0")
        if v > 120:
            raise ValueError("age value is unrealistic")
        return v

    # --- BMI ---
    @validator("bmi")
    def bmi_must_be_realistic(cls, v):
        if v < 10 or v > 90:
            raise ValueError("bmi must be between 10 and 90")
        return v

    # --- CHILDREN ---
    @validator("children")
    def children_cannot_be_negative(cls, v):
        if v < 0:
            raise ValueError("children cannot be negative")
        if v > 20:
            raise ValueError("children value is unrealistic")
        return v

    # --- SEX ONE-HOT ---
    @validator("sex_0", "sex_1")
    def sex_one_hot_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("sex values must be 0 or 1")
        return v

    @validator("sex_1")
    def sex_one_hot_valid(cls, v, values):
        if "sex_0" in values and values["sex_0"] + v != 1:
            raise ValueError("sex one-hot encoding must sum to 1")
        return v

    # --- REGION ONE-HOT ---
    @validator("region_0", 
               "region_1", 
               "region_2", 
               "region_3",
               "bmi_category_normal",
               "bmi_category_underweight",
               "bmi_category_obese",
               "bmi_category_overweight",
               "age_group_0_25",
                "age_group_26_35",
                "age_group_36_45",
                "age_group_46_55",
                "age_group_55_65")
    def region_values_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("region values must be 0 or 1")
        return v

    @validator("region_3")
    def region_one_hot_valid(cls, v, values):
        total = (
            values.get("region_0", 0)
            + values.get("region_1", 0)
            + values.get("region_2", 0)
            + v
        )
        if total != 1:
            raise ValueError("region one-hot encoding must have exactly one 1")
        return v

logging.basicConfig(
    filename="analytics.log",
    format="%(message)s",
    level=logging.INFO
)

#extract for nn only
def extract_features_nn(data: InsuranceData) -> np.ndarray:
    feature_order = [
        "age", "bmi", "bmi_category_normal", "bmi_category_obese",
        "bmi_category_overweight", "bmi_category_underweight", "children",
        "sex_0", "sex_1", "region_0", "region_1", "region_2", "region_3",
        "smoker_0", "smoker_1",
        "age_group_0_25", "age_group_26_35", "age_group_36_45",
        "age_group_46_55", "age_group_55_65"
    ]
    values = [float(getattr(data, f)) for f in feature_order]
    return np.array([values], dtype=np.float32)

def extract_features(data: InsuranceData) -> np.ndarray:
    """
    Convert an InsuranceData object into a numpy feature array of shape (1, n_features)
    """
    feature_order = [
        "age",
        "bmi",
        "bmi_category_normal",
        "bmi_category_obese",
        "bmi_category_overweight",
        "bmi_category_underweight",
        "children",
        "sex_0",
        "sex_1",
        "region_0",
        "region_1",
        "region_2",
        "region_3",
        "smoker_0",
        "smoker_1",
        "age_group_0_25",
        "age_group_26_35",
        "age_group_36_45",
        "age_group_46_55",
        "age_group_55_65",
    ]
    values = [getattr(data, f) for f in feature_order]
    return np.array([values], dtype=float)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API key"
        )

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    end = time.perf_counter()
    process_time = end - start
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def analytics_middleware(request: Request, call_next):

    start = time.perf_counter()
    try:
        body_bytes = await request.body()
        request_size = len(body_bytes)
    except Exception:
        request_size = 0

    request = Request(scope=request.scope, receive=lambda: {"type": "http.request", "body": body_bytes})

    response = await call_next(request)
    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    response_size = len(response_body)

    new_response = Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type
    )

    end = time.perf_counter()
    process_time = end - start
    analytics_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "client_ip": request.client.host if request.client else None,
        "status_code": new_response.status_code,
        "process_time_ms": process_time * 1000,
        "request_size_bytes": request_size,
        "response_size_bytes": response_size,
    }

    logging.info(json.dumps(analytics_data))

    new_response.headers["X-Process-Time"] = str(process_time)
    new_response.headers["X-Request-Size"] = str(request_size)
    new_response.headers["X-Response-Size"] = str(response_size)

    return new_response

@app.get('/')
def main():
    return{'message': 'Welcome to Insurance API'}

@app.post('/predict_charge/rf')
def predict_charge_rf(data : InsuranceData, api_key: str = Security(get_api_key)):
    features = extract_features(data)
    prediction = rf_model.predict(features)

    return {'predicted_price': float(prediction[0])}

@app.post('/predict_charge/xgb')
def predict_charge_xgb(data : InsuranceData, api_key: str = Security(get_api_key)):
    features = extract_features(data)
    prediction = xgb_model.predict(features)

    return {'predicted_price': float(prediction[0])}

'''
@app.post('/predict_charge/nn')
def predict_charge_nn(data: InsuranceData, api_key: str = Security(get_api_key)):
    features = extract_features_nn(data)
    x_tensor = torch.from_numpy(features)

    with torch.no_grad():
        prediction = model(x_tensor).item()

    return {"predicted_price": float(prediction)}
'''