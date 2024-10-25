import sys, cv2, time

try:
    from picamera2 import Picamera2, Preview
    print("Imported picamera2")
except ImportError:
    # do nothing
    print("Did not import picamera2")
    pass

# low code stopwatch that just returns the distance between the last timestamp and now
class Stopwatch():
    def __init__(self):
        self.timestamp = []
    def mark_lap(self):
        self.timestamp.append(time.time())
        if len(self.timestamp) > 1:
            return round((self.timestamp[-1] - self.timestamp[-2])*1000, 2)

timing = False
rpi_status = False
if "-r" in sys.argv:
    rpi_status = True

if "-t" in sys.argv:
    timing = True
    timer = Stopwatch()
    time_records = {}
else:
    timing = False

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
# Load the custom-trained Haar Cascade classifier for face-detection (currently not good)
#face_cascade = cv2.CascadeClassifier("classifier/cascade.xml")

video_capture = None

# Open the webcam (or video file)
if rpi_status:
    video_capture = Picamera2()
    video_capture.start()
else:
    video_capture = cv2.VideoCapture(0)  # Use 0 for webcam. For a video file, provide its path.

faces_in_frame = 0

while True:
    if timing:
        timer.mark_lap()
    frame = None
    # Capture frame-by-frame
    if rpi_status:
        frame = video_capture.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        ret, frame = video_capture.read()

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

    if timing:
        face_detection_time = timer.mark_lap()

    if len(faces) != faces_in_frame:
        print(f"{len(faces)} faces detected")
        faces_in_frame = len(faces)

    if timing:
        try:
            time_records[len(faces)].append(face_detection_time)
        except KeyError:
            time_records[len(faces)] = [face_detection_time]

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
if rpi_status:
    video_capture.close()
else:
    video_capture.release()
cv2.destroyAllWindows()

# compute the statistics for frame by frame face detection
print("=== Compute Time Summary ===")
print("Faces | Mean | Max  | Min  |")
print("------|------|------|------|")
if timing:
    for face_count in time_records:
        mean = sum(time_records[face_count]) / len(time_records[face_count])
        maximum = max(time_records[face_count])
        minimum = min(time_records[face_count])

        print(f"   {int(face_count):02} | {int(mean):04} | {int(maximum):04} | {int(minimum):04} |")


exit()