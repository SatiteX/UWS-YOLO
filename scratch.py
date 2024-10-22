import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO
import tkinter as tk

step = 25
angle = 15

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8m.pt')

# Initialize the Tello drone
tello = Tello()

# tello.set_video_fps("low")
# tello.set_video_resolution('low')
# tello.set_video_bitrate(0)
# tello.set_video_direction(1)

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


# Function to take off
def takeoff(event=None):
    print("Takeoff")
    tello.takeoff()


# Function to land
def land(event=None):
    print("Land")
    tello.land()


# Function to move forward
def move_forward(event=None):
    print("Move Forward")
    tello.move_forward(step)


# Function to move backward
def move_backward(event=None):
    print("Move Backward")
    tello.move_back(step)


# Function to rotate clockwise
def rotate_clockwise(event=None):
    print("Rotate Clockwise")
    tello.rotate_clockwise(angle)


# Function to rotate counterclockwise
def rotate_counterclockwise(event=None):
    print("Rotate Counterclockwise")
    tello.rotate_counter_clockwise(angle)


# Function to move up
def move_up(event=None):
    print("Move Up")
    tello.move_up(step)


# Function to move down
def move_down(event=None):
    print("Move Down")
    tello.move_down(step)


# Function to move left
def move_left(event=None):
    print("Move Left")
    tello.move_left(step)


# Function to move right
def move_right(event=None):
    print("Move Right")
    tello.move_right(step)


def start_stream(event=None):
    print("Streaming")
    while True:
        frame = cap.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Streaming", frame)
        tello.send_command_without_return("command")
        # Break the loop if '5' is pressed
        if cv2.waitKey(1) & 0xFF == ord('5'):
            cv2.destroyAllWindows()
            break
        """if cv2.waitKey(1) & 0xFF == ord('w'):
            tello.move_forward(step)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            tello.move_back(step)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            tello.move_left(step)
        if cv2.waitKey(1) & 0xFF == ord('d'):
            tello.move_right(step)
        if cv2.waitKey(1) & 0xFF == ord('u'):
            tello.move_up(step)
        if cv2.waitKey(1) & 0xFF == ord('j'):
            tello.move_down(step)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            tello.rotate_clockwise(angle)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tello.rotate_counter_clockwise(angle)
        if cv2.waitKey(1) & 0xFF == ord('l'):
            tello.land()"""


# Detection Function:
def detect(event=None):
    print("Detecting in progress")
    while True:
        # Capture the video frame
        frame = cap.frame
        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        tello.send_command_without_return("command")
        # Break the loop if 'x' is pressed
        if cv2.waitKey(1) & 0xFF == ord('5'):
            cv2.destroyAllWindows()
            break


# Create the GUI window
window = tk.Tk()
window.title("DJI Tello Drone Keyboard Control")
window.geometry("400x300")

# Instructions
instructions = tk.Label(window, text="""
Control the Tello Drone with the Keyboard:
W: Move Forward
S: Move Backward
A: Move Left
D: Move Right
Q: Rotate Counterclockwise
E: Rotate Clockwise
Up Arrow: Move Up
Down Arrow: Move Down
T: Takeoff
L: Land
Y: Detect
K: Stream
""")
instructions.pack(pady=12)

# Bind keyboard keys to drone functions
window.bind('<w>', move_forward)
window.bind('<s>', move_backward)
window.bind('<a>', move_left)
window.bind('<d>', move_right)
window.bind('<q>', rotate_counterclockwise)
window.bind('<e>', rotate_clockwise)
window.bind('<Up>', move_up)  # Up arrow key
window.bind('<Down>', move_down)  # Down arrow key
window.bind('<t>', takeoff)
window.bind('<l>', land)
window.bind('<y>', detect)
window.bind('<k>', start_stream)

try:
    window.mainloop()
finally:
    # Clean up resources
    tello.streamoff()
    cv2.destroyAllWindows()
    tello.end()
    print("Cleaned up and disconnected.")
