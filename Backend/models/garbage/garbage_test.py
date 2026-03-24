from ultralytics import YOLO

# Load ONNX model
model = YOLO("best.onnx")

# Run prediction on an image
results = model.predict(source="image_no.jpg", save=True, conf=0.25)

print(results)