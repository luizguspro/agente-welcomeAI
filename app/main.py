from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, List
from uuid import uuid4
from datetime import datetime, timedelta
import os
import io

from .handlers import chat_one_shot, speech_to_text, text_to_speech
from .lead_store import save_lead, list_leads

app = FastAPI(title="Hanna")

# Mount directory for audio files
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Path to frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# In-memory conversation store
_conversations: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
_TIMEOUT = timedelta(minutes=30)

class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    interest: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    audio_url: str


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def _get_history(session_id: str) -> List[Dict[str, str]]:
    now = datetime.utcnow()
    conv = _conversations.get(session_id)
    if not conv or now - conv["last"] > _TIMEOUT:
        _conversations[session_id] = {"history": [], "last": now}
    else:
        conv["last"] = now
    return _conversations[session_id]["history"]

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid4())
    history = _get_history(session_id)
    history.append({"role": "user", "content": req.text})
    reply = await chat_one_shot(req.text, history)
    history.append({"role": "assistant", "content": reply})

    # save lead if provided
    if req.name and req.email and req.interest:
        save_lead(req.name, req.email, req.interest)

    audio_bytes = await text_to_speech(reply)
    audio_name = f"{uuid4()}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_name)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    audio_url = f"/audio/{audio_name}"
    return ChatResponse(session_id=session_id, response=reply, audio_url=audio_url)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid4())
    try:
        while True:
            data = await ws.receive_bytes()
            text = await speech_to_text(data)
            history = _get_history(session_id)
            history.append({"role": "user", "content": text})
            reply = await chat_one_shot(text, history)
            history.append({"role": "assistant", "content": reply})
            audio_bytes = await text_to_speech(reply)
            await ws.send_bytes(audio_bytes)
    except WebSocketDisconnect:
        pass

# Protected leads endpoint
API_KEY = os.getenv("LEADS_API_KEY", "changeme")

def verify_key(key: str = ""):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/leads")
async def get_leads(api_key: str = Depends(verify_key)):
    return list_leads()
