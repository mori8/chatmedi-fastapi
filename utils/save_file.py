from fastapi import UploadFile
from pathlib import Path
import shutil
import uuid


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_file(file: UploadFile) -> str:
    unique_filename = f"{uuid.uuid4()}.png"
    file_location = UPLOAD_DIR / unique_filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return str(file_location)