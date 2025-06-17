import torch
from TTS.api import TTS
from fastapi import FastAPI, Form
import os

from pydantic import BaseModel
import uuid
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

reference_speaker_wav = "audio.wav"

app = FastAPI(title="TTS API com XTTS v2")

@app.get("/")
def read_root():
    return {"message": "API de TTS com XTTS v2 está no ar!"}


class TextInput(BaseModel):
    text: str
    language: str = "fr" 
@app.post("/generate")
def generate_audio(input_data: TextInput):
    text = input_data.text
    lang = input_data.language

    try:
        if not os.path.exists(reference_speaker_wav):
            return {"error": f"Arquivo de referência '{reference_speaker_wav}' não encontrado."}

        filename = f"temp/output_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(".", filename)

        print(f"Gerando áudio para: '{text}'")
        tts.tts_to_file(
            text=text,
            speaker_wav=reference_speaker_wav,
            language=lang,
            file_path=output_path
        )

        print(f"Áudio salvo em {output_path}, reproduzindo...")
        os.system(f'start {output_path}')

        return {"message": "Áudio gerado com sucesso!", "file": filename}

    except Exception as e:
        return {"error": str(e)}



#Where is my book?
#Try another door