import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
import app.main as main


def test_chat(monkeypatch):
    client = TestClient(main.app)

    async def fake_chat_one_shot(prompt, history):
        return "Olá, seja bem-vindo!"

    async def fake_tts(text):
        return b"audio"

    monkeypatch.setattr(main, "chat_one_shot", fake_chat_one_shot)
    monkeypatch.setattr(main, "text_to_speech", fake_tts)

    resp = client.post("/chat", json={"text": "Oi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "Olá, seja bem-vindo!"
    assert data["audio_url"].startswith("/audio/")
