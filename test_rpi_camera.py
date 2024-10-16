from picamera2 import Picamera2, Preview
import cv2

picam2 = Picamera2()
picam2.start()

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow('Camera Module 2', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()

