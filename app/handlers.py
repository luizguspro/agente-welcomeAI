import os
import io
import openai
from elevenlabs import ElevenLabs

SYSTEM_PROMPT = (
    "Você é Hanna, recepcionista virtual do Impact Hub Pedra Branca (Floripa). "
    "Fale de forma cordial, empática, divertida e objetiva, em português do Brasil. "
    "Seu objetivo é acolher visitantes, explicar os serviços do Impact Hub (estações de coworking, salas privativas, salas de reunião, eventos, comunidade global), "
    "coletar dados de contato e oferecer tour presencial ou virtual. Evite jargões técnicos. "
    "Seja sempre proativa em ajudar. Termine cada interação perguntando se pode agendar uma visita ou enviar informações adicionais."
)

openai_client = openai.OpenAI(api_key=os.getenv("keyopenai"))
eleven_client = ElevenLabs(api_key=os.getenv("keyelevelebs"))
VOICE_ID = os.getenv("keyvozelevebs")

async def chat_one_shot(prompt: str, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": prompt}]
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

async def speech_to_text(audio_bytes: bytes) -> str:
    with io.BytesIO(audio_bytes) as f:
        res = eleven_client.speech_to_text.convert(
            model_id="scribe_v1",
            file=f,
            diarize=True,
            file_format="other",
        )
    return res.text

async def text_to_speech(text: str) -> bytes:
    audio_iter = eleven_client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    return b"".join(audio_iter)
