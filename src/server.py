from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import sys
import types
import warnings

# Import your existing churn pipeline components
from inference_engine import churn_management_pipeline, create_module_structure

app = FastAPI(title="Churn Prediction API")

# Define Pydantic model for incoming customer data
class CustomerRecord(BaseModel):
    gender: str
    SeniorCitizen: bool
    Partner: bool
    Dependents: bool
    tenure: int
    PhoneService: bool
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: bool
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Churn prediction endpoint
@app.post("/churn_management")
async def predict_churn(record: CustomerRecord):
    # Convert Pydantic model to DataFrame
    raw_df = pd.DataFrame([record.dict()])

    try:
        # Ensure module structure for unpickling
        create_module_structure()

        # Run the inference pipeline
        result = churn_management_pipeline(raw_df)
        return {
            "churn_probability": result["churn_probability"],
            "is_churning": result["is_churning"],
            "segment": result.get("segment"),
            "report": result.get("report")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
