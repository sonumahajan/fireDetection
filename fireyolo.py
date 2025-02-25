import cv2
import numpy as np
import smtplib
import playsound
import threading

Alarm_Status = False
Email_Status = False
Fire_Reported = 0

# Load YOLO
net = cv2.dnn.readNet("yoloFiles/yolov3.weights", "yoloFiles/yolov3.cfg")  # Load the YOLO weights and config file
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Corrected this line

# Load COCO names for YOLO
with open("yoloFiles/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define function to play alarm sound
def play_alarm_sound_function():
    while True:
        playsound.playsound("alarm-sound.mp3", True)

# Define function to send email
def send_mail_function():
    recipientEmail = "Enter_Recipient_Email"
    recipientEmail = recipientEmail.lower()

    try:
        print("Mail Send")
        # server = smtplib.SMTP("smtp.gmail.com", 587)
        # server.ehlo()
        # server.starttls()
        # server.login(
        #     "Enter_Your_Email (System Email)", "Enter_Your_Email_Password (System Email)"
        # )
        # server.sendmail(
        #     "Enter_Your_Email (System Email)",
        #     recipientEmail,
        #     "Warning A Fire Accident has been reported on ABC Company",
        # )
        # print("sent to {}".format(recipientEmail))
        # server.close()
    except Exception as e:
        print(e)

# Initialize webcam
video = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break

    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangular box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            if classes[class_ids[i]] == "fire":  # Check if the detected object is fire
                Fire_Reported = Fire_Reported + 1
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Fire", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Fire Detection", frame)

    if Fire_Reported >= 1:
        if Alarm_Status == False:
            threading.Thread(target=play_alarm_sound_function).start()
            Alarm_Status = True

        if Email_Status == False:
            threading.Thread(target=send_mail_function).start()
            Email_Status = True

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
video.release()
