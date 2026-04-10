### Brief Description
A complete FastAPI backend that validates user-submitted images using YOLOv26 ONNX object detection models. The backend:
- Receives a POST request with id, image_url and category
- Downloads the image from Supabase Storage via the public URL
- Runs the appropriate ONNX model (pothole.onnx or garbage.onnx or water_clog.onnx)
- Updates the ai_verified and status of the issue in the database
- Returns the result with detection count to the frontend


### Files
- models/ - Contains all 3 Yolo models for detection
- main.py - FastAPI app with POST /api/validate endpoint
- config.py - Loads .env and exposes typed config constants
- models.py - Pydantic schemas for request/response validation
- services/supabase_client.py - Image download (httpx) + DB update (supabase SDK)
- services/yolo_detector.py - ONNX model loading, preprocessing, inference, NMS
- requirements.txt - Pinned Python dependencies


### How it works
- Model caching: ONNX models load lazily on first request per category and stays in memory — subsequent requests are fast.
- Async image download: Uses httpx.AsyncClient for non-blocking image fetching.
- Pure NumPy NMS: No dependency on torchvision or ultralytics — the NMS is a lightweight NumPy implementation.
- Category validation at schema level: Pydantic Literal["pothole", "garbage", "water_clog"] rejects invalid categories before the endpoint code runs.

