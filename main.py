import cv2
import time


# Face detection Model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Eye Detection Model
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

eye_flag = True
show_count = True
blink_count = 0

# Capture the Video Stream
cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()   # Read Frame from the camera
    # img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # First Detect Face and then go for eye detection
    for (x, y, w, h) in faces:
        eye_flag = False   # Set initial value of eye flag
        # Draw rectangle around detected face
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region Of Interest in rectangle of grayscale face image
        # This is for detecting the eyes from the face
        roi_gray = gray[y:y + h, x:x + w]

        # Region Of Interest in rectangle of color face image
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes from the detected face rectangle co-ordinates
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            # Eyes are detected so turn on the flag
            eye_flag = True
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),
                          (255, 255, 0), 2)

    # If we can detect the eyes that means eyes are open
    if eye_flag:
        show_count = True

    # If we have eye_flag = False means we could not detect eye
    # Because eyes are closed
    if show_count and not(eye_flag):
        show_count = False  # Do not show the blink count till open the eye
        blink_count += 1

    # let's display the blink count on image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, 'Blink Count = ' + str(blink_count),
                (25, 50), font, 2, (0, 0, 0), 2)
    # (img, text, position, font_face, font_scale, color, thickness)
    cv2.imshow('Eye Blink Detection!', img)

    # Close the window when q is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
