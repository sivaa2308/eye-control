import cv2
import dlib
import numpy as np
import tensorflow as tf

# Load trained deep learning model
model_path = r"C:\github\eye-control\final_gaze_model.h5"  # Update this path
model = tf.keras.models.load_model(model_path)

# Define label mapping
label_map = {0: "Blink", 1: "Center", 2: "left", 3: "right"}

# Initialize Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = r"c:\final-project-vast\source code\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def extract_eyes(image, expand_ratio=0.2):
    """Extract eye region, expand slightly, and flip if necessary."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if faces:
        face = max(faces, key=lambda r: r.width() * r.height())
        landmarks = predictor(gray, face)

        # Get eye landmarks
        left_eye_pts = [landmarks.part(i) for i in range(36, 42)]
        right_eye_pts = [landmarks.part(i) for i in range(42, 48)]
        all_eye_pts = left_eye_pts + right_eye_pts

        min_x = min(pt.x for pt in all_eye_pts)
        max_x = max(pt.x for pt in all_eye_pts)
        min_y = min(pt.y for pt in all_eye_pts)
        max_y = max(pt.y for pt in all_eye_pts)

        width = max_x - min_x
        height = max_y - min_y
        expand_x = int(width * expand_ratio)
        expand_y = int(height * expand_ratio)

        min_x = max(0, min_x - expand_x)
        max_x = min(image.shape[1], max_x + expand_x)
        min_y = max(0, min_y - expand_y)
        max_y = min(image.shape[0], max_y + expand_y)

        eye_region = image[min_y:max_y, min_x:max_x]
        eye_region = cv2.flip(eye_region, 1)  # Flip to ensure correct left/right
        return eye_region
    return None

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam")
    exit()

print("✅ Press 'q' to exit")

# Track previous predictions for stability
prev_prediction = "Center"

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)  # Flip the entire frame to avoid mirroring issues
    eye_region = extract_eyes(frame, expand_ratio=0.2)

    if eye_region is not None:
        # Preprocess for model input
        eye_region_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
        eye_preprocessed = cv2.resize(eye_region_rgb, (224, 64))
        input_image = eye_preprocessed.astype("float32") / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        # Model prediction
        predictions = model.predict(input_image)
        label_index = np.argmax(predictions)
        confidence = np.max(predictions)

        # Ensure stability for "Left" detection
        if confidence < 0.5:
            predicted_label = prev_prediction  # Keep previous stable prediction
        else:
            predicted_label = label_map.get(label_index, "Unknown")

        # Apply correction for "Left" misclassification
        if predicted_label == "Right" and confidence > 0.6:
            predicted_label = "Left" if prev_prediction == "Left" else "Right"

        prev_prediction = predicted_label  # Save last stable prediction

        # Display the result
        cv2.putText(frame, f"Gaze: {predicted_label} ({confidence:.2f})",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gaze Detection - Presentation Ready", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()