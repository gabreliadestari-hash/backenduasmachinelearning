from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Toko Kue Chatbot API aktif"}

def test_chat_endpoint():
    response = client.post("/chat", json={"message": "halo"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "intent" in data
    assert "confidence" in data
    assert data["intent"] == "salam"

def test_chat_specific_cake():
    response = client.post("/chat", json={"message": "harga red velvet"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "red_velvet"

def test_chat_unknown():
    response = client.post("/chat", json={"message": "xyz123"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] is not None 
