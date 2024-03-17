from ultralytics import YOLO

# Define the path to the dataset.yaml file
dataset_yaml = 'dataset.yaml' 

# Load a pretrained model
model = YOLO('yolov8n.pt')  # Load a pretrained model

# Train the model on your dataset
model.train(data=dataset_yaml, epochs=100)  # Adjust epochs as needed

# Evaluate model performance on the validation set
metrics = model.val()

# Export the model to ONNX format
path = model.export(format="onnx")

# Print out the validation metrics and the export path
print(f"Validation Metrics: {metrics}")
print(f"Model exported to: {path}")

