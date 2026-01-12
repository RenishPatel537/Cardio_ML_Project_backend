from fastapi import FastAPI, HTTPException
from schemas.input_schema import CardioInput
from services.predictor import predict_cardio
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Cardio ML Backend")


origins = [
    "http://localhost:3000",   # React / Next.js
    "http://127.0.0.1:3000",
    "https://cardio-ml-project-frontend-8r11v906g-renish-andanis-projects.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Allowed origins
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],        # Authorization, Content-Type, etc.
)


@app.post("/predict/{model_name}")
def predict(model_name: str, data: CardioInput):

    if model_name not in ["rf", "dt", "lgr", "gb"]:
        raise HTTPException(status_code=400, detail="Invalid model name")

    return predict_cardio(data.dict(), model_name)


## uvicorn main:app --reload
