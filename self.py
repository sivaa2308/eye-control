import cv2
import dlib
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2

# ------------------------------------------------
# 1. Load the Trained Model and Define Labels
# ------------------------------------------------
model_path = "/home/pi/final_gaze_model.h5"  # Your model path
model = tf.keras.models.load_model(model_path)

# Define label mapping – ensure these match your training labels
label_map = {0: "Blink", 1: "Center", 2: "Right", 3: "Left"}

# ------------------------------------------------
# 2. Initialize Dlib Face Detector & Landmark Predictor
# ------------------------------------------------
detector = dlib.get_frontal_face_detector()
predictor_path = "/home/pi/shape_predictor_68_face_landmarks.dat"  # Update if needed
predictor = dlib.shape_predictor(predictor_path)

# ------------------------------------------------
# 3. Setup GPIO for Motor Control
# ------------------------------------------------
GPIO.setwarnings(False)
# Define motor pins (update these if necessary)
L_RPWM, L_LPWM, L_REN, L_LEN = 18, 19, 20, 21  # Left Motor
R_RPWM, R_LPWM, R_REN, R_LEN = 13, 12, 16, 26  # Right Motor

GPIO.setmode(GPIO.BCM)
GPIO.setup([L_RPWM, L_LPWM, L_REN, L_LEN, R_RPWM, R_LPWM, R_REN, R_LEN], GPIO.OUT)

# Create PWM objects for motors at 50Hz
left_pwm_forward = GPIO.PWM(L_RPWM, 50)
left_pwm_reverse = GPIO.PWM(L_LPWM, 50)
right_pwm_forward = GPIO.PWM(R_RPWM, 50)
right_pwm_reverse = GPIO.PWM(R_LPWM, 50)

left_pwm_forward.start(0)
left_pwm_reverse.start(0)
right_pwm_forward.start(0)
right_pwm_reverse.start(0)

# Enable motor drivers
GPIO.output([L_REN, L_LEN, R_REN, R_LEN], GPIO.HIGH)

# ------------------------------------------------
# 4. Blink Detection & Motor Activation Variables
# ------------------------------------------------
blink_start_time = None
blinking = False
long_blink_threshold = 1.0  # Seconds for a long blink
motors_active = False       # Toggle motor activation via long blink

# ------------------------------------------------
# 5. Define the Eye Extraction Function
# ------------------------------------------------
def extract_eyes(image, expand_ratio=0.2):
    """
    Extract the eye region from the image using Dlib landmarks,
    expanding the bounding box slightly.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if faces:
        face = max(faces, key=lambda r: r.width() * r.height())
        landmarks = predictor(gray, face)
        
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
        # Flip horizontally to correct left/right orientation
        eye_region = cv2.flip(eye_region, 1)
        return eye_region
    return None

# ------------------------------------------------
# 6. Define Motor Control Functions
# ------------------------------------------------
def move_forward_wheelchair(speed=80):
    # For forward movement: left motor in reverse, right motor forward
    left_pwm_forward.ChangeDutyCycle(0)
    left_pwm_reverse.ChangeDutyCycle(speed)
    right_pwm_reverse.ChangeDutyCycle(0)
    right_pwm_forward.ChangeDutyCycle(speed)
    print("🚀 Moving Forward")

def turn_left(speed=50):
    left_pwm_forward.ChangeDutyCycle(0)
    right_pwm_forward.ChangeDutyCycle(speed)
    left_pwm_reverse.ChangeDutyCycle(0)
    right_pwm_reverse.ChangeDutyCycle(0)
    print("↩ Turning Left")

def turn_right(speed=50):
    left_pwm_forward.ChangeDutyCycle(speed)
    right_pwm_forward.ChangeDutyCycle(0)
    left_pwm_reverse.ChangeDutyCycle(0)
    right_pwm_reverse.ChangeDutyCycle(0)
    print("↪ Turning Right")

def stop_motors():
    left_pwm_forward.ChangeDutyCycle(0)
    left_pwm_reverse.ChangeDutyCycle(0)
    right_pwm_forward.ChangeDutyCycle(0)
    right_pwm_reverse.ChangeDutyCycle(0)
    print("🛑 Motors Stopped")

# ------------------------------------------------
# 7. Blink Detection for Motor Activation Toggle
# ------------------------------------------------
def detect_long_blink(predicted_label):
    global blink_start_time, blinking, motors_active
    if predicted_label == "Blink":
        if not blinking:
            blink_start_time = time.time()
            blinking = True
    else:
        if blinking:
            blink_duration = time.time() - blink_start_time
            blinking = False
            if blink_duration >= long_blink_threshold:
                motors_active = not motors_active
                if motors_active:
                    print("🟢 Motors Activated by Long Blink")
                else:
                    print("🔴 Motors Deactivated by Long Blink")
                    stop_motors()

# ------------------------------------------------
# 8. Main Function: Gaze Detection + Motor Control
# ------------------------------------------------
def main():
    print("✅ Initializing PiCamera...")
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)
        print("✅ Camera successfully initialized!")
    except Exception as e:
        print(f"❌ Error initializing camera: {e}")
        return
    
    while True:
        try:
            frame = picam2.capture_array()
            frame = cv2.flip(frame, 1)  # Flip frame to avoid mirroring issues
            eye_region = extract_eyes(frame, expand_ratio=0.2)
            
            if eye_region is not None:
                # Preprocess for model input
                eye_region_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
                eye_preprocessed = cv2.resize(eye_region_rgb, (224, 64))
                input_image = eye_preprocessed.astype("float32") / 255.0
                input_image = np.expand_dims(input_image, axis=0)
                
                # Predict gaze direction using the trained model
                predictions = model.predict(input_image)
                label_index = np.argmax(predictions)
                confidence = np.max(predictions)
                
                if confidence < 0.6:
                    predicted_label = "Center"
                else:
                    predicted_label = label_map.get(label_index, "Unknown")
                
                # Toggle motor activation via long blink
                detect_long_blink(predicted_label)
                
                # Control motors if they are activated
                if motors_active:
                    if predicted_label == "Left":
                        turn_left(50)
                    elif predicted_label == "Right":
                        turn_right(50)
                    elif predicted_label == "Center":
                        move_forward_wheelchair(60)
                    elif predicted_label == "Blink":
                        stop_motors()
                
                # Display prediction on frame for debugging
                cv2.putText(frame, f"Gaze: {predicted_label} ({confidence:.2f})", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show the frame (if a display is connected)
            cv2.imshow("Gaze Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            break
    
    picam2.stop()
    stop_motors()
    cv2.destroyAllWindows()
    GPIO.cleanup()

# ------------------------------------------------
# 9. Run the Main Function
# ------------------------------------------------
if _name_ == "_main_":
    main()