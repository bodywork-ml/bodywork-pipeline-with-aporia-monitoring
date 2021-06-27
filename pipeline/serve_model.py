"""
This module loads a pre-trained SciKit-Learn model, configure Aporia for
monitoring the model, and defines a web service using FastAPI to serve
predictions, using Pydantic for data validation.
"""
import logging
import os
import sys
from typing import Dict, Optional
from urllib.request import urlopen

import aporia
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, status
from numpy import array, floating
from pydantic import BaseModel
from sklearn.base import BaseEstimator


CATEGORY_TP_INTEGER_MAP = {"c0": 0, "c1": 1, "c2": 2}
MODEL_URL = (
    "http://bodywork-pipeline-with-aporia-monitoring"
    ".s3.eu-west-2.amazonaws.com/models/model.joblib"
)

app = FastAPI(debug=False)


class FeatureDataInstance(BaseModel):
    """Define JSON data schema for prediction requests."""

    id: str
    f1: float
    f2: str


class Prediction(BaseModel):
    """Define JSON data schema for prediction response."""

    y_pred: float


@app.post("/api/v1/predict", status_code=status.HTTP_200_OK, response_model=Prediction)
async def predict(data: FeatureDataInstance) -> Dict[str, floating]:
    """Generate predictions for data sent to the /api/v1/predict route."""
    try:
        f2_encoded = CATEGORY_TP_INTEGER_MAP[data.f2]
        X = array([[data.f1, f2_encoded]])

        prediction = {"y": float(model.predict(X))}

        if aporia_client is not None:
            aporia_client.log_prediction(
                id=data.id,
                raw_inputs={
                    "F_1": data.f1,
                    "F_2": f2_encoded,
                },
                features={
                    "F_1": data.f1,
                    "F_2": data.f2,
                },
                predictions=prediction,
            )

        return prediction
    except KeyError as e:
        msg = f"Unknown category provided for f2 - {e}"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=msg
        )
    except Exception as e:
        msg = f"Could not generate prediction - {e}"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg
        )


def load_model(url: str) -> BaseEstimator:
    """Download model from AWS S3 and load it into memory."""
    try:
        model = joblib.load(urlopen(url))
        log.info("ML model loaded into memory.")
        return model
    except Exception as e:
        msg = f"Could not fetch and/or load model - {e}"
        log.error(msg)
        raise RuntimeError


def configure_logger() -> logging.Logger:
    """Configure a logger that will write to stdout."""
    log_handler = logging.StreamHandler(sys.stdout)
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s"
    )
    log_handler.setFormatter(log_format)
    log = logging.getLogger(__name__)
    log.addHandler(log_handler)
    log.setLevel(logging.INFO)
    return log


def configure_aporia_monitoring() -> Optional[aporia.model.Model]:
    """Configure the Aporia client API for ML model monitoring."""
    try:
        token = os.environ["APORIA_TOKEN"]
        host = os.environ.get("APORIA_HOST")
        environment = os.environ.get("APORIA_ENVIRONMENT", "local-dev")
        
        # IMPORTANT: Consider removing verbose=True in production.
        aporia.init(token=token, host=host, environment=environment, verbose=True)

        return aporia.Model(model_id="bodywork-test-8wxi", model_version="v1")
    except KeyError:
        msg = "Could not find required APORIA_TOKEN or APORIA_MODEL_ID environment variable."
        log.warning(msg)
        return None
    except Exception as e:
        msg = f"Could not configure Aporia monitoring client - {e}"
        log.warning(msg)
        return None


log = configure_logger()
model = load_model(MODEL_URL)
aporia_client = configure_aporia_monitoring()


if __name__ == "__main__":
    log.info("Starting prediction server.")
    uvicorn.run(app, host="0.0.0.0", workers=1)
