import cv2
import torch
import os
from datetime import datetime
from playsound import playsound  # For playing sound alerts
import threading  # For controlling audio playback duration

# Load the trained YOLOv5 model
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="\\yolov5\\runs\\train\\exp3\\weights\\best.pt",
)  # Adjust with path to best.pt

# Create the 'alert_fire' folder if it doesn't exist
alert_folder = "alert_fire"
if not os.path.exists(alert_folder):
    os.makedirs(alert_folder)

# Path to the alert sound file
alert_sound_path = "alarm-sound.wav"

# Start webcam feed
cap = cv2.VideoCapture(0)  # 0 is the default webcam on most systems

# Set a confidence threshold
confidence_threshold = 0.7  # Only consider detections with confidence >= 0.7

# Variable to track if the alert sound has been played
alert_played = False


# Function to play sound for some seconds
def play_sound_for_seconds(sound_path):
    playsound(sound_path)
    # Stop the sound after 2 seconds
    threading.Timer(2.0, stop_sound).start()


# Function to stop the sound
def stop_sound():
    global alert_played
    alert_played = False  # Reset the alert flag


while True:
    ret, frame = cap.read()  # Capture frame by frame
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Filter out detections with low confidence
    filtered_results = results.pandas().xyxy[0][
        results.pandas().xyxy[0]["confidence"] >= confidence_threshold
    ]

    # Check if fire is detected with confidence >= threshold
    if not filtered_results.empty and "fire" in filtered_results["name"].values:
        # Generate a unique filename using the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(alert_folder, f"fire_alert_{timestamp}.jpg")

        # Render results (bounding boxes, labels, etc.)
        rendered_frame = results.render()[0]

        # Save the rendered frame (with bounding boxes) as an image
        cv2.imwrite(filename, rendered_frame)
        print(
            f"Fire detected with confidence >= {confidence_threshold}! Image saved as {filename}"
        )

        # Play the alert sound if it hasn't been played already
        if not alert_played:
            threading.Thread(
                target=play_sound_for_seconds, args=(alert_sound_path,)
            ).start()
            alert_played = True  # Prevent the sound from playing repeatedly
    else:
        # Reset the alert flag if no fire is detected
        alert_played = False

    # Display the resulting frame with predictions
    cv2.imshow("Fire Detection", results.render()[0])

    # Press 'q' to exit the video loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
