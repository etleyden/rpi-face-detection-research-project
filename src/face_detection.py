import sys
import cv2

try:
    from picamera2 import Picamera2, Preview
    print("Imported picamera2")
except ImportError:
    # do nothing
    print("Did not import picamera2")
    pass

if len(sys.argv) < 2:
    print("Usage: python face_detection.py <rpi_status>")
    exit()

rpi_status = int(sys.argv[1])


print(rpi_status)
# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
# Load the custom-trained Haar Cascade classifier for face-detection (currently not good)
#face_cascade = cv2.CascadeClassifier("classifier/cascade.xml")

video_capture = None
# Open the webcam (or video file)
if rpi_status == 1:
    video_capture = Picamera2()
    video_capture.start()
else:
    video_capture = cv2.VideoCapture(0)  # Use 0 for webcam. For a video file, provide its path.

while True:
    frame = None
    # Capture frame-by-frame
    if rpi_status == 1:
        frame = video_capture.capture_array()
    else:
        ret, frame = video_capture.read()

    if rpi_status == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to grayscale (Haar Cascade works with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        print(f"{len(faces)} faces detected")

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
if rpi_status == 1:
    video_capture.close()
else:
    video_capture.release()
cv2.destroyAllWindows()