from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO('/home/stan/Desktop/pystuff/neck_damaged11.pt')

# Export the model to ONNX format
model.export(format='onnx', imgsz=[640, 640])  # You can adjust the image size as needed

print("Model exported to ONNX format successfully!")


