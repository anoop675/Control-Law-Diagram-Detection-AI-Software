import cv2
from google.colab import drive
from google.colab.patches import cv2_imshow
drive.mount('/content/gdrive')

ROOT_DIR = '/content/gdrive/My Drive/data'

!pip install ultralytics

import os
from ultralytics import YOLO

'''
Use for model training and comment the rest
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
results = model.train(data=os.path.join(ROOT_DIR, "google_colab_config.yaml"), epochs=65)  # train the model
print(results)  # Shows where results and weights are saved
'''

#Load trained model
model = YOLO("runs/detect/train4/weights/best.pt")
# Use the model

#!scp -r /content/runs '/content/gdrive/My Drive/data'

# Run detection correctly (ensuring it returns `YOLO.Result`)
results = model.predict("temp9.jpg")  # Run inference properly

# Load the input image
image = cv2.imread("temp9.jpg")

# Process results properly
for result in results:
    boxes = result.boxes  # List of detected bounding boxes

    for box in boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to int
        class_id = int(box.cls)  # Class index
        conf = box.conf.item()  # Confidence score

        if conf >= 0.7:
          # Draw bounding box
          color = (0, 255, 0)  # Green box
          cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

          # Put class label text above the bounding box
          label = f"{model.names[class_id]} ({conf:.2f})"
          print(label)
          cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the image with detections
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image
cv2.imwrite("output_with_boxes.jpg", image)
