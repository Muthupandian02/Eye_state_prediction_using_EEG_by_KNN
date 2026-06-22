import joblib
import pickle
import os
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

app = FastAPI(title="Eye State Prediction using EEG")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "pickle_files", "knn_model.pkl"))

try:
    with open(os.path.join(BASE_DIR, "pickle_files", "scaler.pkl"), "rb") as file1:
        scaler = pickle.load(file1)
except Exception as e:
    print("Error loading scaler:", e)

# Define input schema
class EEG(BaseModel):
    AF3: float
    F7: float
    F3: float
    FC5: float
    T7: float
    P7: float
    O1: float
    O2: float
    P8: float
    T8: float
    FC6: float
    F4: float
    F8: float
    AF4: float

# Set up templates directory using dynamic path
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Home route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="home.html"
)

# Description route
@app.get("/description", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="description.html"
)

# Prediction route
@app.get("/prediction", response_class=HTMLResponse)
def prediction_page(request: Request):
    return templates.TemplateResponse(
        request = request,
        name="prediction.html"
        )

@app.post("/prediction", response_class=HTMLResponse)
def home(
    request: Request,
    AF3: float = Form(...),
    F7: float = Form(...),
    F3: float = Form(...),
    FC5: float = Form(...),
    T7: float = Form(...),
    P7: float = Form(...),
    O1: float = Form(...),
    O2: float = Form(...),
    P8: float = Form(...),
    T8: float = Form(...),
    FC6: float = Form(...),
    F4: float = Form(...),
    F8: float = Form(...),
    AF4: float = Form(...)
    ):

    values = EEG(AF3=AF3, F7=F7, F3=F3, FC5=FC5, T7=T7, P7=P7, O1=O1, O2=O2,
                      P8=P8, T8=T8, FC6=FC6, F4=F4, F8=F8, AF4=AF4)
    
    df=pd.DataFrame([values.model_dump()])

    df = df.drop(columns=['O1','P8','F8','AF4'])

    df_scaled=scaler.transform(df)

    prediction = model.predict(df_scaled)

    result = "Eyes are Closed" if prediction[0] == 1 else "Eyes are opened"

    return templates.TemplateResponse(
    request=request,
    name="prediction.html",
    context={"result": result}
    )