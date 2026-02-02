import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

SUPABASE_URL: str = get_env("SUPABASE_URL")
SUPABASE_SERVICE_KEY: str = get_env("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET: str = os.getenv("SUPABASE_BUCKET", "upload_images")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        ext = file.filename.split(".")[-1]
        storage_path = f"{file_id}.{ext}"

        file_bytes = await file.read()

        # Upload file (raises error automatically if it fails)
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": file.content_type},
        )

        # Get public URL
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
        if isinstance(public_url, dict):
            public_url = public_url.get("publicUrl")

        # Insert metadata
        db_response = supabase.table("Image").insert({
            "id": file_id,
            "image_path": public_url,
            "uploaded_at": datetime.utcnow().isoformat()
        }).execute()

        return db_response.data[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
