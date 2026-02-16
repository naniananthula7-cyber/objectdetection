import cv2
import csv
import os
from ultralytics import YOLO
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# --- Step 1: Authenticate Google Drive ---
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Opens browser for authentication
drive = GoogleDrive(gauth)

# --- Step 2: Load YOLOv8 model ---
model = YOLO("yolov8n.pt")

# --- Step 3: Start webcam ---
cap = cv2.VideoCapture(0)

# Dictionary to store counts (unique detections)
object_counts = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Get detected class names
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Count only once per object type
            if cls_name not in object_counts:
                object_counts[cls_name] = 1
            else:
                object_counts[cls_name] += 1

    # Annotate detections
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Step 4: Save counts to CSV locally ---
csv_filename = "object_counts.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Object", "Count"])
    for obj, count in object_counts.items():
        writer.writerow([obj, count])

print(f"Saved counts to {csv_filename}")

# --- Step 5: Upload CSV to Google Drive ---
file_drive = drive.CreateFile({'title': csv_filename})
file_drive.SetContentFile(csv_filename)
file_drive.Upload()
print("CSV uploaded to Google Drive successfully!")