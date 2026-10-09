"""
RiskLens – unit tests for FastAPI Signal Store.
Run with: pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["engine"] == "RiskLens"

def test_get_signals_endpoint():
    response = client.get("/signals")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "signals" in data
    assert isinstance(data["signals"], list)

def test_get_signals_filter():
    response = client.get("/signals?ticker=SIVB&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "signals" in data
    for sig in data["signals"]:
        assert sig["entity"]["ticker"] == "SIVB"

def test_signals_by_ticker():
    # SIVB is present in the SVB replay dataset
    response = client.get("/signals/SIVB")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "SIVB"
    assert "signals" in data

def test_signals_by_ticker_not_found():
    response = client.get("/signals/UNKNOWN_XYZ_999")
    assert response.status_code == 404
