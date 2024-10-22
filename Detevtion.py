import cv2
import numpy as np
from djitellopy import Tello
import time

# Initialize the Tello drone
tello = Tello()
tello.connect()

# Print battery percentage for safety
print(f"Battery Level: {tello.get_battery()}%")

# Start the video stream
tello.streamon()
cap = tello.get_frame_read()

# Define HSV color range for the object to track (in this case, red)
lower_red = np.array([0, 0, 100])
upper_red = np.array([100, 100, 255])

# Take off
tello.takeoff()
time.sleep(3)  # Wait for the drone to stabilize

try:
    while True:
        # Get the video frame from Tello
        frame = cap.frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert BGR frame to HSV
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the red color
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Find the largest contour (assuming it's the object we want to track)
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            # Draw a circle around the object
            if radius > 10:  # Adjust the minimum radius to avoid noise
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

                # Calculate the center of the frame
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2

                # Calculate offset from the center
                offset_x = int(x) - frame_center_x
                offset_y = int(y) - frame_center_y

                # Control the drone based on offset
                if offset_x < -50:  # Object is to the left of center
                    tello.move_left(20)
                elif offset_x > 50:  # Object is to the right of center
                    tello.move_right(20)

                if offset_y < -50:  # Object is above the center
                    tello.move_up(20)
                elif offset_y > 50:  # Object is below the center
                    tello.move_down(20)

        # Display the original frame
        cv2.imshow('Tello Tracking', hsv)
        tello.send_command_without_return("command")
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted manually")

finally:
    # Land the drone and release resources
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()
    tello.end()

"""import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8m.pt')

# Connect to the Tello drone
tello = Tello()

# Loop until the drone is successfully connected
connected = False
while not connected:
    try:
        tello.connect()
        connected = True
        print("Successfully connected to Tello drone!")
        print("Battery %:", tello.get_battery())
    except Exception as e:
        print(f"Failed to connect to Tello drone: {e}")
        print("Retrying in 3 seconds...")
        time.sleep(3)  # Wait for 3 seconds before retrying

# Start the video stream
stream = False
while not stream:
    try:
        tello.streamon()
        stream = True
        print("Successfully Streaming from Tello drone!")
    except Exception as e:
        print(f"Failed to stream from Tello drone: {e}")
        print("Retrying in 3 seconds...")
        time.sleep(3)  # Wait for 3 seconds before retrying

# Get the video stream from the drone
# cap = tello.get_frame_read()
# time.sleep(3)

getframe = False
while not getframe:
    try:
        cap = tello.get_frame_read()
        getframe = True
        print("Successfully Reading frame from Tello drone!")
    except Exception as e:
        print(f"Failed to read frame from Tello drone: {e}")
        print("Retrying in 3 seconds...")
        time.sleep(3)  # Wait for 3 seconds before retrying

while True:
    # Capture the video frame
    frame = cap.frame
    stream = cap.frame
    # Convert the frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stream = cv2.cvtColor(stream, cv2.COLOR_BGR2RGB)
    # Run YOLOv8 inference on the frame and filter for class 0 (person)
    results = model.predict(source=frame, classes=[0])

    # Draw bounding boxes for people
    for result in results:
        for det in result.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())  # Convert to integers
            conf = det.conf.item()  # Convert tensor to a Python scalar (float)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Person: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Detection", frame)
    cv2.imshow("Tello Video Stream", stream)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close any open windows
#cap.release()
tello.streamoff()
cv2.destroyAllWindows()
"""