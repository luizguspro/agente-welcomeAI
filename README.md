# Hanna - Impact Hub Pedra Branca

Agente virtual de recepção usando FastAPI, OpenAI GPT-4o e ElevenLabs.

## Execução local

Configure as variáveis de ambiente `keyopenai`, `keyelevelebs`, `keyvozelevebs` e opcionalmente `LEADS_API_KEY`.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Abra `http://localhost:8000/` para acessar o front‑end.

Endpoints principais:
- `POST /chat` – conversa textual e retorna áudio.
- `WS /ws` – chat em tempo real via voz.
- `GET /health` – health-check.
- `GET /leads` – lista de leads (requer `LEADS_API_KEY`).

## Docker

```bash
docker build -t hanna .
docker run -p 8000:8000 hanna
```
Depois de iniciado, acesse `http://localhost:8000/`.
