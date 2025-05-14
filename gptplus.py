import cv2
import numpy as np
import dlib
from math import hypot

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def eye_aspect_ratio(eye_points, landmarks):
    # Compute the eye aspect ratio (EAR)
    p1 = landmarks.part(eye_points[0])
    p2 = landmarks.part(eye_points[1])
    p3 = landmarks.part(eye_points[2])
    p4 = landmarks.part(eye_points[3])
    p5 = landmarks.part(eye_points[4])
    p6 = landmarks.part(eye_points[5])
    
    # Calculate the distances
    vertical1 = hypot(p2.x - p6.x, p2.y - p6.y)
    vertical2 = hypot(p3.x - p5.x, p3.y - p5.y)
    horizontal = hypot(p1.x - p4.x, p1.y - p4.y)
    
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def get_gaze_direction(eye_points, landmarks, gray):
    # Extract the eye region from landmarks
    pts = np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in eye_points], np.int32)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [pts], 255)
    eye_region = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Get the bounding box of the eye region
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    roi = eye_region[min_y:max_y, min_x:max_x]

    if roi.size == 0:
        return "center", 0.5

    # Apply Gaussian blur and thresholding to highlight the pupil
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    _, thresh = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    height, width = thresh.shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cx = width // 2  # default center
    if contours:
        # Choose the largest contour (assumed to be the pupil)
        pupil = max(contours, key=cv2.contourArea)
        M = cv2.moments(pupil)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
    
    ratio = cx / width
    # Adjusted thresholds:
    # If ratio < 0.45, more dark area on left → user is looking LEFT.
    # If ratio > 0.65, user is looking RIGHT.
    # Otherwise, CENTER.
    if ratio < 0.45:
        return "left", ratio
    elif ratio > 0.65:
        return "right", ratio
    else:
        return "center", ratio

def draw_text(frame, text, pos, color):
    cv2.putText(frame, text, pos, font, 1, color, 2)

# Colors for display (LEFT: Blue, RIGHT: Red, CENTER: Green, BLINK: Yellow)
colors = {
    "left": (255, 0, 0),
    "right": (0, 0, 255),
    "center": (0, 255, 0),
    "blink": (0, 255, 255)
}

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("c:/final-project-vast/source code/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

# Blink threshold: if EAR is below this value, the eye is considered closed.
BLINK_THRESHOLD = 0.25  # Typical values range ~0.2-0.3 for open eyes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect and convert to grayscale
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Process only the largest detected face
    largest_face = max(faces, key=lambda r: r.width() * r.height()) if faces else None

    if largest_face is not None:
        try:
            landmarks = predictor(gray, largest_face)
            # Compute EAR for each eye using the EAR formula
            ear_right = eye_aspect_ratio([36, 37, 38, 39, 40, 41], landmarks)
            ear_left  = eye_aspect_ratio([42, 43, 44, 45, 46, 47], landmarks)
            avg_ear = (ear_right + ear_left) / 2.0

            if avg_ear < BLINK_THRESHOLD:
                state = "blink"
                state_color = colors["blink"]
            else:
                # Compute gaze direction using the gaze ratio for each eye
                gaze_right, ratio_right = get_gaze_direction([36, 37, 38, 39, 40, 41], landmarks, gray)
                gaze_left, ratio_left = get_gaze_direction([42, 43, 44, 45, 46, 47], landmarks, gray)
                avg_ratio = (ratio_right + ratio_left) / 2.0
                
                if gaze_right == gaze_left:
                    state = gaze_right
                else:
                    state = "center"
                state_color = colors.get(state, (255, 255, 255))
                # Display average ratio for debugging
                draw_text(frame, f"Ratio: {avg_ratio:.2f}", (largest_face.left(), largest_face.top()-30), (255,255,255))
            
            draw_text(frame, state.upper(), (largest_face.left(), largest_face.top()-10), state_color)
            cv2.rectangle(frame, (largest_face.left(), largest_face.top()),
                          (largest_face.right(), largest_face.bottom()), (255,0,0), 2)
        except Exception as e:
            print("Error processing face:", e)

    cv2.imshow("Gaze Detection", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
