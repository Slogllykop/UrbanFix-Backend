### Brief Description
A complete FastAPI backend that validates user-submitted images using YOLOv26 ONNX object detection models. The backend:
- Receives a POST request with id, image_url and category
- Downloads the image from Supabase Storage via the public URL
- Runs the appropriate ONNX model (pothole_best.onnx or garbage_best.onnx)
- Updates the validate column in Supabase to "valid" (object found) or "invalid" (no object)
- Returns the result with detection count to the frontend


### Files
- main.py - FastAPI app with POST /api/validate endpoint
- config.py - Loads .env and exposes typed config constants
- models.py - Pydantic schemas for request/response validation
- services/supabase_client.py - Image download (httpx) + DB update (supabase SDK)
- services/yolo_detector.py - ONNX model loading, preprocessing, inference, NMS
- requirements.txt - Pinned Python dependencies


### How it works
- Model caching: ONNX models load lazily on first request per category and stay in memory — subsequent requests are fast.
- Async image download: Uses httpx.AsyncClient for non-blocking image fetching.
- Pure NumPy NMS: No dependency on torchvision or ultralytics — the NMS is a lightweight NumPy implementation.
- Category validation at schema level: Pydantic Literal["pothole", "garbage"] rejects invalid categories before the endpoint code runs.


### Refer to HowToRun.txt for a quick description on how to run this project otherwise you can refer to README.md.

