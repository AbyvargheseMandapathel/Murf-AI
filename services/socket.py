import uuid
import os

async def save_audio_chunk(audio_data, session_id):
    #save an incoming audio chunk to a file
    os.makedirs("recordings", exist_ok=True)
    
    #use sessionId for the filename to keep all chunks from the same session together
    file_path =f"recordings/{session_id}.ogg"
    
    #append audio data to the file
    with open(file_path, "ab") as f:
        f.write(audio_data)
    
    return file_path    