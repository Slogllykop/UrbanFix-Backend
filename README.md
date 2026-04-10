# UrbanFix Validation Backend

A **FastAPI** backend that validates user-submitted images (potholes / garbage) using **YOLOv8 ONNX** models. The API receives an image URL and category from the frontend, runs object detection, and updates the validation status in a **Supabase** database.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Database Setup (Supabase)](#database-setup-supabase)
3. [Environment Configuration](#environment-configuration)
4. [Installation](#installation)
5. [Running the Server](#running-the-server)
6. [API Reference](#api-reference)
7. [Making Requests](#making-requests)

---

## Architecture Overview

```
Frontend                     Backend                         Supabase
   │                            │                               │
   │  POST /api/validate        │                               │
   │  {id, image_url, category} │                               │
   │ ─────────────────────────► │                               │
   │                            │  1. Fetch image via URL       │
   │                            │ ────────────────────────────► │
   │                            │  ◄──── image bytes ────────── │
   │                            │                               │
   │                            │  2. Run YOLO ONNX model       │
   │                            │     (pothole / garbage)       │
   │                            │                               │
   │                            │  3. UPDATE reports SET        │
   │                            │     validate = 'valid'/'invalid'
   │                            │ ────────────────────────────► │
   │                            │                               │
   │  ◄── {status, detections}  │                               │
   │                            │                               │
```

---

## Database Setup (Supabase)

### Step 1: Create the table

Go to your Supabase dashboard --> Create table with the name `issues` and add the following columns:

| Column      | Type   | Description                                         |
| ----------- | ------ | --------------------------------------------------- |
| `id`        | UUID | Primary key – Auto Generate Random                  |
| `status` | TEXT   | Status of the current issue (verified/rejected)          |
| `ai_verified`  | BOOL   | Whether issue has been verified by AI or not                           |
| `priority_score`  | NUMERIC   | Priority count of the issue    |
| `category`        | TEXT | Issue category (pothole, garbage, water_clog)                  |
| `image_url`        | TEXT | Public URL of image stored in the bucket                  |

### Step 2: Verify your Storage bucket

Make sure:
1. Your images are uploaded to a **Supabase Storage bucket**.
2. The bucket is either **public**, or you're using signed URLs.
3. The `image_url` column stores the **full public URL** to the image.

Example public URL format:
```
https://<project-id>.supabase.co/storage/v1/object/public/<bucket-name>/<file-path>
```

---

## Environment Configuration

A `.env` example file has been provided where you can add your own supabase environment variables:


| Variable                 | Description                                           | Default   |
| ------------------------ | ----------------------------------------------------- | --------- |
| `SUPABASE_URL`           | Your Supabase project URL                             | —         |
| `SUPABASE_KEY`           | Service role API key                                  | —         |
| `TABLE_NAME`             | Database table containing reports                     | `issues` |
| `CONFIDENCE_THRESHOLD`   | Minimum confidence for a detection to count           | `0.5`     |
| `IOU_THRESHOLD`          | IoU threshold for Non-Maximum Suppression             | `0.45`    |

---

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/Scavenblaze/UrbanFix
cd UrbanFix
```

### Step 2: Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```
---

## Running the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload
```

Open **http://localhost:8000** in your browser

Interactive API docs are available at: **http://localhost:8000/docs**

---

## API Reference

### `GET /`

Health check endpoint.

**Response:**
```json
{ "status": "ok", "service": "UrbanFix Validation API" }
```

---

### `POST /api/validate`

Validate an image using YOLO object detection.

**Request Body:**
```json
{
    "id": "uuid of the image",
    "image_url": "public url of image",
    "category": "pothole/garbage/water_clog"
}
```

| Field       | Type   | Required | Description                                    |
| ----------- | ------ | -------- | ---------------------------------------------- |
| `id`        | string    | ✅       | Row ID in the database to update               |
| `image_url` | string | ✅       | Full public URL of the image in Supabase Storage |
| `category`  | string | ✅       | `"pothole"` or `"garbage"` or `"water_clog"`                      |

**Success Response (200):**
```json
{
    "id": "83b2988b-0f5b-41db-8f36-62ba413baf4e",
    "status": "valid",
    "message": "Image validated as 'valid' with 3 detection(s).",
    "detections": 3
}
```

**Error Responses:**

| Status | Reason                                              |
| ------ | --------------------------------------------------- |
| 400    | Image URL is unreachable or file is not a valid image |
| 422    | Invalid request body (missing field, bad category)   |
| 500    | Model file missing or detection / database error     |

---

## Making Requests

### Using cURL

```bash
curl -X POST http://localhost:8000/api/validate \
  -H "Content-Type: application/json" \
  -d '{
    "id": "[Generated-uuid-string]",
    "image_url": "[image-public-url]",
    "category": "[pothole or garbage or water_clog]"
  }'
```

### Using Python (httpx)

```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/validate",
    json={
        "id": "[Generated-uuid-string]",
        "image_url": "[image-public-url]",
        "category": "[pothole or garbage or water_clog]",
    },
)
print(response.json())
```

### Using JavaScript (fetch)

```javascript
const response = await fetch("http://localhost:8000/api/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        "id": "[Generated-uuid-string]",
        "image_url": "[image-public-url]",
        "category": "[pothole or garbage or water_clog]",
    }),
});

const data = await response.json();
console.log(data);
```

---
