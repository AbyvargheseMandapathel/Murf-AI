from fastapi import FastAPI, HTTPException, UploadFile, File
import requests
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import assemblyai as aai
import json


from pydantic import BaseModel

class LLMQuery(BaseModel):
    text: str


load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
aai.settings.api_key = ASSEMBLYAI_API_KEY


app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join("app/static", "index.html"))



class TextInput(BaseModel):
    text: str

MURF_API_KEY = os.getenv("MURF_API_KEY")




@app.post("/generate-audio/")
def generate_audio(input: TextInput):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "api-key": MURF_API_KEY
    }

    body = {
        "text": input.text,
        "voice_id": "en-US-amara",
        "style":"Narration"
    }
    

    response = requests.post("https://api.murf.ai/v1/speech/generate", json=body, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    result = response.json()
    # print(result)
    return {"audio_url": result.get("audioFile")}

UPLOAD_DIR = os.path.join("app", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-echo/")
async def upload_echo(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)
    return JSONResponse({
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content)
    })


@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    try:
        # Read file bytes
        audio_bytes = await file.read()

        # Use AssemblyAI's Transcriber
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_bytes)

        return {"text": transcript.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/tts/echo")
async def tts_echo(file: UploadFile = File(...)):
    try:
        # Save uploaded audio temporarily
        filepath = os.path.join(UPLOAD_DIR, file.filename)
        with open(filepath, "wb") as f:
            f.write(await file.read())

        # Transcribe with AssemblyAI (new SDK)
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(filepath)  # file path works here

        if transcript.status != aai.TranscriptStatus.completed:
            raise HTTPException(status_code=500, detail="Transcription failed")

        text_to_speak = transcript.text

        # Send to Murf for TTS
        murf_url = "https://api.murf.ai/v1/speech/generate"
        headers = {
            "accept": "application/json",
            "api-key": MURF_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "voice_id": "en-US-natalie",  # pick your voice
            "text": text_to_speak,
            "format": "mp3",
            "sampleRate": 44100
        }

        murf_response = requests.post(murf_url, json=payload, headers=headers)
        if murf_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Murf API error: {murf_response.text}"
            )

        murf_data = murf_response.json()
        audio_url = murf_data.get("audioFile")
        if not audio_url:
            raise HTTPException(status_code=500, detail="No audio URL returned from Murf")

        return {
            "audio_url": audio_url,
            "transcript": text_to_speak
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@app.post("/llm/query")
async def query_llm_audio(file: UploadFile = File(...)):
    try:
        # 1️⃣ Save uploaded audio
        audio_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        # 2️⃣ Transcribe with AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path)
        if transcript.status != aai.TranscriptStatus.completed:
            raise HTTPException(status_code=500, detail="Transcription failed")
        user_text = transcript.text

        # 3️⃣ Send transcription to Gemini API
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [
                {
                    "parts": [{"text": user_text}]
                }
            ]
        }
        llm_response = requests.post(url, headers=headers, json=payload)
        if llm_response.status_code != 200:
            raise HTTPException(status_code=500, detail=llm_response.text)
        llm_data = llm_response.json()

        try:
            generated_text = llm_data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise HTTPException(status_code=500, detail="Unexpected Gemini response format")

        # 4️⃣ Send Gemini output to Murf for TTS (split if > 3000 chars)
        murf_url = "https://api.murf.ai/v1/speech/generate"
        headers_murf = {
            "accept": "application/json",
            "api-key": MURF_API_KEY,
            "Content-Type": "application/json"
        }

        if len(generated_text) > 3000:
            # Optional: split into multiple audio segments
            chunks = [generated_text[i:i+3000] for i in range(0, len(generated_text), 3000)]
            audio_urls = []
            for chunk in chunks:
                payload_murf = {
                    "voice_id": "en-US-natalie",
                    "text": chunk,
                    "format": "mp3",
                    "sampleRate": 44100
                }
                murf_resp = requests.post(murf_url, json=payload_murf, headers=headers_murf)
                if murf_resp.status_code != 200:
                    raise HTTPException(status_code=500, detail=murf_resp.text)
                audio_urls.append(murf_resp.json().get("audioFile"))
            return {"transcript": user_text, "llm_text": generated_text, "audio_urls": audio_urls}
        else:
            payload_murf = {
                "voice_id": "en-US-natalie",
                "text": generated_text,
                "format": "mp3",
                "sampleRate": 44100
            }
            murf_resp = requests.post(murf_url, json=payload_murf, headers=headers_murf)
            if murf_resp.status_code != 200:
                raise HTTPException(status_code=500, detail=murf_resp.text)
            murf_data = murf_resp.json()
            return {
                "transcript": user_text,
                "llm_text": generated_text,
                "audio_url": murf_data.get("audioFile")
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))