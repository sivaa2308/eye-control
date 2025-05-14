import cv2
import dlib
import os
import time

# Load Dlib's face detector & landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("c:/final-project-vast/source code/shape_predictor_68_face_landmarks.dat")

# Define dataset path (update folder name as needed: 'right', 'left', 'center', or 'blink')
save_path = "model_train/validate/center"
os.makedirs(save_path, exist_ok=True)

# Set persistent image counter based on existing files in the folder
existing_files = [f for f in os.listdir(save_path) if f.startswith("right_") and f.endswith(".jpg")]
if existing_files:
    image_counter = max([int(f.split('_')[1].split('.')[0]) for f in existing_files]) + 1
else:
    image_counter = 1

# Open the laptop webcam (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access laptop webcam")
    exit()

def extract_eyes(image, expand_ratio=0.2):
    """
    Extract a slightly larger combined eye region using detected facial landmarks.
    expand_ratio: How much to expand the bounding box (default: 20% extra in each direction).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)

        # Get points for left eye (indices 36-41) and right eye (indices 42-47)
        left_eye_pts = [landmarks.part(i) for i in range(36, 42)]
        right_eye_pts = [landmarks.part(i) for i in range(42, 48)]
        
        # Combine both sets of points
        all_eye_pts = left_eye_pts + right_eye_pts
        min_x = min(pt.x for pt in all_eye_pts)
        max_x = max(pt.x for pt in all_eye_pts)
        min_y = min(pt.y for pt in all_eye_pts)
        max_y = max(pt.y for pt in all_eye_pts)
        
        # Expand bounding box size by the given ratio
        width = max_x - min_x
        height = max_y - min_y
        expand_x = int(width * expand_ratio)
        expand_y = int(height * expand_ratio)
        
        # Adjust bounding box and ensure it stays within image bounds
        min_x = max(0, min_x - expand_x)
        max_x = min(image.shape[1], max_x + expand_x)
        min_y = max(0, min_y - expand_y)
        max_y = min(image.shape[0], max_y + expand_y)
        
        return image[min_y:max_y, min_x:max_x]
    
    return None  # Return None if no face is detected

# Use a timer to capture images every 0.5 seconds
last_capture_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image from webcam")
            break

        # Display the webcam feed
        cv2.imshow("Webcam", frame)

        # Check if it's time to capture the eye image (every 0.5 seconds)
        if time.time() - last_capture_time >= 0.5:
            last_capture_time = time.time()
            eye_region = extract_eyes(frame, expand_ratio=0.2)  # Increase the region by 20%
            if eye_region is not None:
                # Resize to the required resolution (224x64)
                eye_resized = cv2.resize(eye_region, (224, 64))
                # Save the image to the specified folder
                image_filename = os.path.join(save_path, f"right_{image_counter}.jpg")
                cv2.imwrite(image_filename, eye_resized)
                print(f"✅ Image saved: {image_filename}")
                image_counter += 1
            else:
                print("⚠️ No face detected in the frame.")

        # Allow the user to exit by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Stopping image capture...")

finally:
    cap.release()
    cv2.destroyAllWindows()
