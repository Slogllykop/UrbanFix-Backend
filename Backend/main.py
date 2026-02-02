import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from supabase import create_client, Client
import uuid
import io
from PIL import Image

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "upload_images")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        #Create unique file id for the uploaded image
        file_id = str(uuid.uuid4())
        storage_path = f"{file_id}.jpg"

        file_bytes = await file.read()

        img = Image.open(io.BytesIO(file_bytes))
        
        compressed_img = io.BytesIO()
        
        img.save(compressed_img, format="JPEG", optimize=True, quality=85)
        
        compressed_img.seek(0)

        #Upload file to storage
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path,
            file=compressed_img.read(),
            file_options={"content-type": "image/jpeg"},
        )

        #Generate file url
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
        if isinstance(public_url, dict):
            public_url = public_url.get("publicUrl")

        #Insert to db
        db_response = supabase.table("Image").insert({
            "image_path": public_url,
        }).execute()

        return db_response.data[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
