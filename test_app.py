"""
Unit and Integration Tests for Flask Prediction API and Endpoints.
"""

import json
from app import app

def test_endpoints():
    client = app.test_client()
    
    print("[*] Testing GET /health...")
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.status_code}"
    print("    Health Response:", res.get_json())
    
    print("[*] Testing POST /predict with user request format...")
    payload = {
        "study_time": 5,
        "absences": 2,
        "previous_score": 75
    }
    res = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200, f"Predict failed: {res.status_code}, {res.get_data(as_text=True)}"
    data = res.get_json()
    print("    Prediction Response:", data)
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in ["Pass", "Fail"]
    assert isinstance(data["confidence"], (float, int))
    
    print("[*] Testing POST /predict with failing student profile...")
    fail_payload = {
        "study_time": 0.5,
        "absences": 22,
        "previous_score": 35
    }
    res_fail = client.post("/predict", data=json.dumps(fail_payload), content_type="application/json")
    assert res_fail.status_code == 200
    data_fail = res_fail.get_json()
    print("    Prediction (Fail Profile):", data_fail)
    
    print("[*] Testing GET / dashboard...")
    res_ui = client.get("/")
    assert res_ui.status_code == 200
    assert "Student Performance" in res_ui.get_data(as_text=True)
    
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    test_endpoints()
