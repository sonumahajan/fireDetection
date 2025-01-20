import cv2
import numpy as np
import smtplib
import playsound
import threading
import time

Alarm_Status = False
Email_Status = False
Fire_Reported = 0


# Function to play the alarm sound in a separate thread
def play_alarm_sound_function():
    while True:
        # playsound.playsound("alarm-sound.mp3", True)
        playsound.playsound("alarm-sound.wav", True)


# Function to send an email notification
def send_mail_function():
    recipientEmail = "Enter_Recipient_Email"  # Replace with actual email
    recipientEmail = recipientEmail.lower()

    try:
        pass
        # server = smtplib.SMTP("smtp.gmail.com", 587)
        # server.ehlo()
        # server.starttls()
        # server.login(
        #     "Enter_Your_Email", "Enter_Your_Email_Password"
        # )  # Replace with actual email credentials
        # server.sendmail(
        #     "Enter_Your_Email",  # Replace with actual sender email
        #     recipientEmail,
        #     "Warning: A fire accident has been reported.",
        # )
        # print("Email sent to {}".format(recipientEmail))
        # server.close()
    except Exception as e:
        print("Error sending email:", e)


# Function to detect fire using color segmentation
def detect_fire(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define fire-like color ranges (red and yellow)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # Create masks for red and yellow colors
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine the two masks
    mask = cv2.bitwise_or(mask_red, mask_yellow)

    return mask


# Initialize video capture from the webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame for processing
    frame = cv2.resize(frame, (960, 540))

    # Apply Gaussian blur to smooth the image
    blur = cv2.GaussianBlur(frame, (21, 21), 0)

    # Detect fire using the fire detection function
    fire_mask = detect_fire(frame)

    # Count the number of non-zero (fire-like) pixels in the mask
    fire_pixels = cv2.countNonZero(fire_mask)

    if fire_pixels > 15000:  # Adjust this threshold based on your environment
        Fire_Reported += 1

    # Display the fire detection output
    output = cv2.bitwise_and(frame, frame, mask=fire_mask)
    cv2.imshow("Fire Detection", output)

    # Trigger alarm and email if fire is reported
    if Fire_Reported >= 1:
        if not Alarm_Status:
            threading.Thread(
                target=play_alarm_sound_function
            ).start()  # Start alarm sound in a new thread
            Alarm_Status = True

        if not Email_Status:
            threading.Thread(
                target=send_mail_function
            ).start()  # Start email notification in a new thread
            Email_Status = True

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close any OpenCV windows
video.release()
cv2.destroyAllWindows()
