# utils.py
import os
from fastapi import UploadFile
from app.config import settings
from app.logger import logger

def save_uploaded_file(file: UploadFile) -> str:
    file_location = os.path.join(settings.UPLOAD_DIR, file.filename)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    with open(file_location, "wb") as f:
        f.write(file.file.read())

    logger.info(f"File saved: {file_location}")
    return file_location