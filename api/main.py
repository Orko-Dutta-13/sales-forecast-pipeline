from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import duckdb
import os


# --------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Sales Forecast API",
    description="Predicts retail sales using XGBoost trained on Ecuadorian store data",
    version="1.0.0"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODELS_DIR = r"C:\Users\oorko\OneDrive\Documents\projects\Sales_Forecasting\models"

DB_PATH = r"C:\Users\oorko\OneDrive\Documents\projects\Sales_Forecasting\data\sales_forecast.db"


# --------------------------------------------------
# Load model files
# --------------------------------------------------

print("Loading models...")

model = joblib.load(
    os.path.join(MODELS_DIR, "xgb_model.pkl")
)

le_family = joblib.load(
    os.path.join(MODELS_DIR, "le_family.pkl")
)

le_store = joblib.load(
    os.path.join(MODELS_DIR, "le_store_type.pkl")
)

feature_cols = joblib.load(
    os.path.join(MODELS_DIR, "feature_cols.pkl")
)


# --------------------------------------------------
# Load store metadata from DuckDB
# --------------------------------------------------

conn = duckdb.connect(DB_PATH)

stores_df = conn.execute("""
    SELECT
        store_nbr,
        type AS store_type,
        cluster
    FROM raw_stores
""").fetchdf()
conn.close()

print("Models and store metadata loaded successfully")
print(f"Features expected: {feature_cols}")


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def health_check():

    return {
        "status": "API is running",
        "model": "XGBoost",
        "features": len(feature_cols)
    }


# --------------------------------------------------
# Prediction request structure
# --------------------------------------------------

class PredictionRequest(BaseModel):

    store_nbr: int
    family: str
    date: str
    onpromotion: int
    oil_price: float
    is_national_holiday: int

    # Can be missing
    transactions: float | None = None

    sales_lag_7: float
    sales_lag_14: float
    sales_lag_28: float
    sales_rolling_7day_avg: float
    sales_rolling_28day_avg: float


# --------------------------------------------------
# Single prediction
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    # ----------------------------------------------
    # 1. Parse date
    # ----------------------------------------------

    try:
        date = pd.Timestamp(request.date)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )


    # IMPORTANT:
    # pandas:
    # Monday = 0 ... Sunday = 6
    #
    # DuckDB DAYOFWEEK used during training:
    # Sunday = 0 ... Saturday = 6

    day_of_week = (date.dayofweek + 1) % 7

    week_of_year = int(date.isocalendar().week)

    is_weekend = 1 if day_of_week in (0, 6) else 0

    quarter = (date.month - 1) // 3 + 1


    # ----------------------------------------------
    # 2. Find store metadata
    # ----------------------------------------------

    store_row = stores_df[
        stores_df["store_nbr"] == request.store_nbr
    ]

    if store_row.empty:

        raise HTTPException(
            status_code=404,
            detail=f"Store {request.store_nbr} not found"
        )


    store_type = store_row["store_type"].iloc[0]

    cluster = int(
        store_row["cluster"].iloc[0]
    )


    # ----------------------------------------------
    # 3. Encode family
    # ----------------------------------------------

    family_name = request.family.strip().upper()

    try:

        family_encoded = le_family.transform(
            [family_name]
        )[0]

    except ValueError:

        valid = list(le_family.classes_)

        raise HTTPException(
            status_code=400,
            detail=f"Unknown family. Valid options: {valid}"
        )


    # ----------------------------------------------
    # 4. Encode store type
    # ----------------------------------------------

    try:

        store_type_encoded = le_store.transform(
            [store_type]
        )[0]

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail=f"Store type encoding error for type: {store_type}"
        )


    # ----------------------------------------------
    # 5. Handle transactions
    # ----------------------------------------------

    transactions = (
        request.transactions
        if request.transactions is not None
        else np.nan
    )


    # ----------------------------------------------
    # 6. Create feature values
    # ----------------------------------------------

    feature_values = {

        "store_nbr": request.store_nbr,

        "family": family_encoded,

        "year": date.year,

        "month": date.month,

        "day": date.day,

        "day_of_week": day_of_week,

        "week_of_year": week_of_year,

        "quarter": quarter,

        "is_weekend": is_weekend,

        "onpromotion": request.onpromotion,

        "store_type": store_type_encoded,

        "cluster": cluster,

        "oil_price": request.oil_price,

        "is_national_holiday":
            request.is_national_holiday,

        "transactions": transactions,

        "sales_lag_7":
            request.sales_lag_7,

        "sales_lag_14":
            request.sales_lag_14,

        "sales_lag_28":
            request.sales_lag_28,

        "sales_rolling_7day_avg":
            request.sales_rolling_7day_avg,

        "sales_rolling_28day_avg":
            request.sales_rolling_28day_avg,
    }


    # ----------------------------------------------
    # 7. Correct feature order
    # ----------------------------------------------

    input_df = pd.DataFrame(
        [feature_values]
    )[feature_cols]


    # ----------------------------------------------
    # 8. Predict
    # ----------------------------------------------

    prediction = model.predict(
        input_df
    )[0]

    prediction = max(
        0,
        float(prediction)
    )


    # ----------------------------------------------
    # 9. Return prediction
    # ----------------------------------------------

    return {

        "store_nbr":
            request.store_nbr,

        "family":
            family_name,

        "date":
            request.date,

        "store_type":
            store_type,

        "predicted_sales":
            round(prediction, 2)
    }


# --------------------------------------------------
# Batch prediction request
# --------------------------------------------------

class BatchRequest(BaseModel):

    predictions: list[PredictionRequest]


# --------------------------------------------------
# Batch prediction endpoint
# --------------------------------------------------

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):

    results = []

    for item in request.predictions:

        result = predict(item)

        results.append(result)

    return {
        "predictions": results,
        "count": len(results)
    }