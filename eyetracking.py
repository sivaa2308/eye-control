import cv2
import numpy as np
import dlib
from math import hypot

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    
    # Calculate horizontal and vertical distances
    ver_length = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])
    hori_length = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])
    
    # Avoid division by zero
    if ver_length == 0:
        return 0
    
    return hori_length / ver_length

def draw_text(frame, text, position, color=(255, 0, 0)):
    cv2.putText(frame, text, position, font, 1, color, 2)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("c:/final-project-vast/source code/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

BLINK_THRESHOLD = 5.7  # Adjust based on your use case
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Calculate blinking ratios for both eyes
        left_eye_ratio = blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        # Average blinking ratio
        blink_average = (left_eye_ratio + right_eye_ratio) / 2

        if left_eye_ratio > BLINK_THRESHOLD and right_eye_ratio <= BLINK_THRESHOLD:
            draw_text(frame, "Right Eye Close", (50, 150))
        elif right_eye_ratio > BLINK_THRESHOLD and left_eye_ratio <= BLINK_THRESHOLD:
            draw_text(frame, "Left Eye Close", (50, 110))
        elif blink_average > BLINK_THRESHOLD:
            draw_text(frame, "Both Eyes Closed", (50, 50))
        else:
            draw_text(frame, "Eyes Open", (50, 200), color=(0, 255, 0))

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
