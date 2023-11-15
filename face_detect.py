import cv2 as cv
import dlib
import numpy as np

# Open the built-in camera (default camera index is usually 0)

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv.VideoCapture(0)



# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces_rect = detector(gray)
    
    # Check if the frame was read successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Display the frame
    for face in faces_rect:
        x,y,w,h = face.left(), face.top(), face.width(), face.height()
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    #cv.imshow("Camera Feed", frame)
    
        for face in faces_rect:
            # Get facial landmarks
            landmarks = landmark_predictor(gray, face)

            # Extract the coordinates of the left and right eyes
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            nose = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(28, 36)])
            mouse = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
            # Draw polygons around the eyes, nose, mouth
            cv.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=2)
            cv.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=2)
            cv.polylines(frame, [nose], isClosed= True, color=(0, 255, 0), thickness=2)
            cv.polylines(frame, [mouse], isClosed= True, color=(0, 255, 0), thickness=2)

    cv.imshow("Eye Tracking", frame)
    # Break the loop if the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv.destroyAllWindows()
