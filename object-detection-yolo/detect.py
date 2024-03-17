from ultralytics import YOLO
import cv2

# Define the path to the ONNX model and a test image and a image from the val dataset
model_path = 'runs/detect/train/weights/best.onnx'
val_image_path = 'datasets/val/images/fc8c5bf7-5033-4dff-b4a0-cf7f3163d492.jpg'
test_image_path = 'test/test.png'

# Load the ONNX model
model = YOLO(model_path)

# Load the images with OpenCV
val_image= cv2.imread(val_image_path)
test_image = cv2.imread(test_image_path)


# Check if images is loaded properly
if  val_image is None:
    print(f"Failed to load image {val_image_path}")
    exit(1)
elif test_image is None:
    print(f"Failed to load image {test_image_path}")
    exit(1)


# Predict using the model
result_val = model(val_image)
result_test = model(test_image)

# Display the test image with the detections
result_val[0].show()  
result_test[0].show()  



