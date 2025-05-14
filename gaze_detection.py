import cv2
import dlib
import numpy as np
import tensorflow as tf
import os

# ------------------------------
# Load the trained model from the specified location (Windows)
# ------------------------------
model_path = r"C:\github\eye-control\latest_newmodel.h5"  # Update with your model location
print("📢 Loading model from:", model_path)
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Label mapping (adjust if needed)
label_map = {0: "Blink", 1: "Center", 2: "Left", 3: "Right"}

# ------------------------------
# Initialize Dlib's face detector and landmark predictor
# ------------------------------
detector = dlib.get_frontal_face_detector()
predictor_path = r"c:\final-project-vast\source code\shape_predictor_68_face_landmarks.dat"  # Update if necessary
predictor = dlib.shape_predictor(predictor_path)

def extract_eyes(image, expand_ratio=0.2):
    """
    Extract and slightly expand the combined eye region from the image using dlib's landmarks.
    Returns the cropped eye region, or None if no face is detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if faces:
        # Process the largest detected face
        face = max(faces, key=lambda r: r.width() * r.height())
        landmarks = predictor(gray, face)
        # Get landmarks for both eyes (indices 36-41 for right, 42-47 for left)
        left_eye_pts = [landmarks.part(i) for i in range(36, 42)]
        right_eye_pts = [landmarks.part(i) for i in range(42, 48)]
        all_eye_pts = left_eye_pts + right_eye_pts
        min_x = min(pt.x for pt in all_eye_pts)
        max_x = max(pt.x for pt in all_eye_pts)
        min_y = min(pt.y for pt in all_eye_pts)
        max_y = max(pt.y for pt in all_eye_pts)
        # Expand bounding box by the given ratio
        width = max_x - min_x
        height = max_y - min_y
        expand_x = int(width * expand_ratio)
        expand_y = int(height * expand_ratio)
        min_x = max(0, min_x - expand_x)
        max_x = min(image.shape[1], max_x + expand_x)
        min_y = max(0, min_y - expand_y)
        max_y = min(image.shape[0], max_y + expand_y)
        return image[min_y:max_y, min_x:max_x]
    return None

def resize_with_aspect_ratio(image, target_width, target_height):
    """
    Resize the image while preserving the aspect ratio and pad with black borders if necessary.
    """
    h, w = image.shape[:2]
    target_aspect = target_width / target_height
    current_aspect = w / h

    if current_aspect > target_aspect:
        # Width is dominant: scale based on target width
        new_width = target_width
        new_height = int(new_width / current_aspect)
    else:
        # Height is dominant: scale based on target height
        new_height = target_height
        new_width = int(current_aspect * new_height)

    resized = cv2.resize(image, (new_width, new_height))

    # Calculate padding
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded

# ------------------------------
# Open the laptop webcam
# ------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam")
    exit()

print("✅ Press 'q' to exit")

# ------------------------------
# Main Loop: Capture, Process, and Run Inference
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    # Extract the eye region from the frame
    eye_region = extract_eyes(frame, expand_ratio=0.2)
    
    if eye_region is not None:
        # Convert the eye region from BGR to RGB (if your model was trained on RGB images)
        eye_region_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
        
        # Resize the eye region to the required resolution (224x64) with aspect ratio preservation
        eye_preprocessed = resize_with_aspect_ratio(eye_region_rgb, 224, 64)
        
        # Normalize pixel values and expand dimensions for model input
        input_image = eye_preprocessed.astype("float32") / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        
        # Make a prediction using the model
        predictions = model.predict(input_image)
        label_index = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_label = label_map.get(label_index, "Unknown")
        
        # Overlay the predicted label and confidence on the frame
        cv2.putText(frame, f"Gaze: {predicted_label} ({confidence:.2f})", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with prediction
    cv2.imshow("Gaze Detection - Test", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
