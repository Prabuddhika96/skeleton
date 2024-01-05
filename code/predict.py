from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('best.pt')
model.predict(source='2.mp4', show=False, conf=0.5, save=False)

# Perform object detection and get predictions
# results = model.predict(source='shuttle_19.jpg', show=False, conf=0.5, save=False)

# Retrieve bounding box coordinates of detected objects
# bounding_boxes = results.xyxy[0].numpy().tolist()

# print(bounding_boxes)

# Print bounding box coordinates for each object
# for bbox in bounding_boxes:
#     class_id, confidence, x_min, y_min, x_max, y_max = bbox[:6]  # Extracting coordinates

#     # Bounding box coordinates
#     bounding_box_coordinates = {
#         "class_id": class_id,
#         "confidence": confidence,
#         "x_min": x_min,
#         "y_min": y_min,
#         "x_max": x_max,
#         "y_max": y_max
#     }

#     print("Bounding Box Coordinates:", bounding_box_coordinates)
