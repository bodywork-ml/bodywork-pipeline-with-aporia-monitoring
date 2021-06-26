"""Functional tests for the prediction service."""
from typing import Any, Dict
from unittest.mock import patch, MagicMock

from pipeline.serve_model import app

from fastapi.testclient import TestClient
from pytest import approx, fixture

test_client = TestClient(app)


@fixture(scope="function")
def valid_payload() -> Dict[str, Any]:
    return {"id": "001", "f1": 0.5, "f2": "c1"}


def test_service_return_predictions_for_valid_payloads(valid_payload: Dict[str, Any]):
    response = test_client.post(url="api/v1/predict", json=valid_payload)
    assert response.status_code == 200
    assert response.json()["y_pred"] == approx(0.6412045368458209)


def test_service_raises_http_422_for_invalid_category(valid_payload: Dict[str, Any]):
    valid_payload["f2"] = "c4"
    response = test_client.post(url="api/v1/predict", json=valid_payload)
    assert response.status_code == 422
    assert "Unknown category provided for f2" in response.json()["detail"]


def test_service_raises_http_422_for_invalid_payload(valid_payload: Dict[str, Any]):
    del valid_payload["f2"]
    response = test_client.post(url="api/v1/predict", json=valid_payload)
    assert response.status_code == 422


@patch("pipeline.serve_model.model.predict", side_effect=Exception)
def test_service_raises_http_500_for_prediction_exception(
    mock_model: MagicMock,
    valid_payload: Dict[str, Any]
):
    mock_model.side_effect = Exception
    response = test_client.post(url="api/v1/predict", json=valid_payload)
    assert response.status_code == 500
