from fastapi import FastAPI, HTTPException
import requests
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


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
