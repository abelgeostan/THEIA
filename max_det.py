import cv2
from ultralytics import YOLO
import time
from picamera2 import Picamera2
from collections import Counter
import os
import threading
import queue
from gpiozero import Button
#this is the main code as of 8 may 2025
# GPIO pin setup for buttons
DAMAGE_DETECTION_PIN = 17  # GPIO17 (pin 11)
VOLUME_UP_PIN = 22        # GPIO22 (pin 15)
VOLUME_DOWN_PIN = 23      # GPIO23 (pin 16)

# Initialize buttons
damage_detection_button = Button(DAMAGE_DETECTION_PIN, pull_up=True)
volume_up_button = Button(VOLUME_UP_PIN, pull_up=True)
volume_down_button = Button(VOLUME_DOWN_PIN, pull_up=True)

# Audio setup
audio_queue = queue.Queue()
current_volume = 50  # Default volume level

# Detection states
damage_detection_mode = False
counting_active = False
totalsum = 0
seen_objects = set()
detection_buffer = []
frames_collected = 0
last_audio_time = 0
last_detection_time = 0

# Function to detect the correct audio device index (0, 1, or 2)
def detect_audio_device():
    test_wav = "/tmp/test_audio.wav"
    
    # Generate a tiny silent test file
    os.system(f'ffmpeg -f lavfi -i anullsrc -t 0.1 -acodec pcm_s16le -ar 16000 -ac 1 {test_wav} 2>/dev/null')
    
    # Try each possible index (0, 1, 2)
    for index in [0, 1, 2]:
        cmd = f'aplay -D plughw:{index},0 {test_wav} 2>/dev/null'
        exit_code = os.system(cmd)
        
        if exit_code == 0:
            os.remove(test_wav)
            return index
    
    os.remove(test_wav)
    return None

# Detect the correct index ONCE at startup
AUDIO_DEVICE_INDEX = detect_audio_device()
print(f"[DEBUG] Using audio device index: {AUDIO_DEVICE_INDEX}")

def set_volume(volume):
    """Set the volume for the specific audio device"""
    global current_volume
    current_volume = max(0, min(100, volume))  # Clamp between 0-100
    
    if AUDIO_DEVICE_INDEX is not None:
        os.system(f'amixer -D hw:{AUDIO_DEVICE_INDEX} set Master {current_volume}% unmute')
    else:
        os.system(f'amixer set Master {current_volume}% unmute')
    print(f"Volume set to {current_volume}%")

def increase_volume():
    """Increase volume by 5%"""
    set_volume(current_volume + 5)

def decrease_volume():
    """Decrease volume by 5%"""
    set_volume(current_volume - 5)

def start_damage_detection():
    """Handle damage detection button press"""
    global damage_detection_mode
    if not damage_detection_mode:
        damage_detection_mode = True
        speech("Please unfold your note and keep it steady")
        threading.Thread(target=damage_detection_sequence).start()

def speech_worker():
    while True:
        text = audio_queue.get()
        wav_file = "/tmp/tts.wav"
        os.system(f'pico2wave -w {wav_file} "{text}"')
        
        # Use detected index or fallback to default
        if AUDIO_DEVICE_INDEX is not None:
            os.system(f'aplay -D plughw:{AUDIO_DEVICE_INDEX},0 {wav_file}')
        else:
            os.system(f'aplay {wav_file}')  # System default
        
        os.remove(wav_file)
        audio_queue.task_done()

def speech(text):
    audio_queue.put(text)

# Start worker thread once
threading.Thread(target=speech_worker, daemon=True).start()

# Initialize with default volume
set_volume(current_volume)

# Load models
model_path = '/home/stan/Desktop/pystuff/neck11.onnx'  # Main currency detection model
model = YOLO(model_path, task='detect')

damage_model_path = '/home/stan/Desktop/pystuff/neck_damaged11.onnx'  # Damage detection model
damage_model = YOLO(damage_model_path, task='detect') if os.path.exists(damage_model_path) else None

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
picam2.set_controls({"AfMode": 2})  # Continuous Autofocus

# Create window
cv2.namedWindow('Currency Detection', cv2.WINDOW_NORMAL)
cv2.moveWindow('Currency Detection', 100, 100)
cv2.resizeWindow('Currency Detection', 800, 600)

# Directories to save frames
original_dir = "/home/stan/Desktop/pystuff/detected_frames"
annotated_dir = "/home/stan/Desktop/pystuff/annotated_frames"
os.makedirs(original_dir, exist_ok=True)
os.makedirs(annotated_dir, exist_ok=True)

# Mapping classes to spoken feedback
class_to_text = {
    "100_new_damaged": "Damaged 100 rupee note detected",
    "100_old_damaged": "Damaged 100 rupee note detected",
    "10_new_damaged": "Damaged 10 rupee note detected",
    "10_old_damaged": "Damaged 10 rupee note detected",
    "20_new_damaged": "Damaged 20 rupee note detected",
    "20_old_damaged": "Damaged 20 rupee note detected",
    "50_new_damaged": "Damaged 50 rupee note detected",
    "50_old_damaged": "Damaged 50 rupee note detected",
    "500_new": "500 rupee note detected",
    "500_folded": "500 rupee note detected",
    "200_new": "200 rupee note detected",
    "200_new_folded": "200 rupee note detected",
    "100_new": "100 rupee note detected",
    "100_new_folded": "100 rupee note detected",
    "50_new": "50 rupee note detected",
    "50_new_folded": "50 rupee note detected",
    "20_new": "20 rupee note detected",
    "20_new_folded": "20 rupee note detected",
    "10_new": "10 rupee note detected",
    "10_new_folded": "10 rupee note detected",
    "50_old": "50 rupee note detected",
    "50_old_folded": "50 rupee note detected",
    "20_old": "20 rupee note detected",
    "20_old_folded": "20 rupee note detected",
    "10_old": "10 rupee note detected",
    "10_old_folded": "10 rupee note detected",
    "screen_image": "Screen image detected, be cautious"
}

def damage_check(image):
    currency_detected = False
    damaged_flag = False
    feedback_text = None
    
    if damage_model:
        results = damage_model(image, conf=0.5)
        
        if results and len(results[0]) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = damage_model.names[class_id]
                
                if "damaged" in class_name.lower():
                    damaged_flag = True
                
                if class_name in class_to_text:
                    feedback_text = class_to_text[class_name]
                    currency_detected = True
    
    return feedback_text, currency_detected, damaged_flag

def check_with_normal_model(image):
    """Check the note with the normal model to get denomination"""
    results = model(image, conf=0.7)
    denomination = None
    
    if results and len(results[0]) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            
            # Skip damaged classes and screen images
            if "damaged" in class_name.lower() or "screen" in class_name.lower():
                continue
                
            # Extract denomination from class name (first part before underscore)
            denomination = class_name.split('_')[0]
            if denomination.isdigit():
                return int(denomination), class_name
    
    return None, None

def damage_detection_sequence():
    global damage_detection_mode
    
    # Countdown
    for i in range(3, 0, -1):
        speech(str(i))
        time.sleep(1)
    
    speech("Capturing image")
    
    # Capture frame
    frame_bgr = picam2.capture_array()
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
    
    # Save the captured frame
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    damage_frame_path = os.path.join(original_dir, f"damage_check_{timestamp}.jpg")
    cv2.imwrite(damage_frame_path, frame)
    
    # Check for damage
    feedback_text, currency_detected, damaged_flag = damage_check(frame)
    

    if damaged_flag:
        speech(feedback_text)
    else:
        # Check with normal model to get denomination
        denomination, class_name = check_with_normal_model(frame)
        if denomination is not None:
            speech(f"{denomination} rupee note detected. Note is not damaged.")
        else:
            speech("Could not determine denomination")
    
    
    # Wait before returning to normal mode
    time.sleep(3)
    damage_detection_mode = False

def fast_draw(frame, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls)
        label = f"{class_names[cls_id]}"
        
        # Use red color for screen images
        if "screen_image" in label.lower():
            color = (0, 0, 255)  # Red
            thickness = 3
        else:
            color = (0, 255, 0)  # Green
            thickness = 2
            
        # Add tracking ID if available
        if box.id is not None:
            label += f" (ID:{int(box.id)})"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return frame

# Attach event listeners
damage_detection_button.when_pressed = start_damage_detection
volume_up_button.when_pressed = increase_volume
volume_down_button.when_pressed = decrease_volume

# Main loop
try:
    speech("Welcome to THEIA")
    swaha = 0
    
    while True:
        if damage_detection_mode:
            # Skip processing while in damage detection mode
            time.sleep(0.1)
            continue
            
        frame_bgr = picam2.capture_array()
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
        current_time = time.time()

        if counting_active and (current_time - last_audio_time >= 10):
            print("Counting mode terminated due to inactivity\n")
            speech(f"Counting mode terminated, total sum is {totalsum} rupees")
            totalsum = 0
            counting_active = False
            seen_objects.clear()
            detection_buffer = []
            frames_collected = 0
            last_detection_time = 0

        if frames_collected > 0 and (current_time - last_detection_time >= 3):
            print(f"Resetting frame collection after 3 seconds of no detections. Collected {frames_collected} frames.")
            detection_buffer = []
            frames_collected = 0

        # Normal detection mode
        results = model.track(frame, persist=True, conf=0.70, verbose=False)
        
        if results and results[0].boxes.id is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            swaha=0
            # Save original frame
            original_frame_path = os.path.join(original_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(original_frame_path, frame)
            print(f"Saved original frame: {original_frame_path}")

            # Generate and save annotated frame
            annotated_frame = fast_draw(frame.copy(), results[0].boxes, model.names)
            annotated_frame_path = os.path.join(annotated_dir, f"annotated_{timestamp}.jpg")
            cv2.imwrite(annotated_frame_path, annotated_frame)
            print(f"Saved annotated frame: {annotated_frame_path}")

            frame_class_ids = []

            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                
                # Check for screen images first
                if "screen_image" in class_name.lower():
                    if (class_id, "warning") not in seen_objects:
                        speech("Screen image detected, be cautious")
                        seen_objects.add((class_id, "warning"))
                        last_audio_time = current_time
                    continue
                
                try:
                    object_id = int(box.id)
                    frame_class_ids.append((class_id, object_id))
                except:
                    print("Tracking ID issue")
                    continue

            if frame_class_ids:
                detection_buffer.append(frame_class_ids)
                frames_collected += 1
                last_detection_time = current_time

            if frames_collected >= 5:
                if detection_buffer:
                    all_class_ids = [class_id for frame_detections in detection_buffer for (class_id, _) in frame_detections]
                    if all_class_ids:
                        counter = Counter(all_class_ids)
                        most_common_class_id, count = counter.most_common(1)[0]
                        class_name = model.names[most_common_class_id]

                        for frame_detections in reversed(detection_buffer):
                            for class_id, obj_id in frame_detections:
                                if class_id == most_common_class_id:
                                    unique_object = (obj_id, class_name)
                                    break
                            else:
                                continue
                            break

                        if unique_object not in seen_objects and class_name in class_to_text:
                            feedback_text = class_to_text[class_name]
                            print(feedback_text)
                            speech(feedback_text)

                            content = class_name.split('_')
                            rupees = int(content[0])
                            checker = content[-1]

                            if checker != "folded":
                                if not counting_active:
                                    speech("New counting mode started")
                                    counting_active = True
                                totalsum += rupees
                                print(f"Total sum: {totalsum}\n")
                            else:
                                print("Folded note detected")

                            seen_objects.add(unique_object)
                            last_audio_time = current_time

                detection_buffer = []
                frames_collected = 0
                last_detection_time = 0
        else:
            annotated_frame = frame
            swaha += 1
            if swaha == 10:
                seen_objects.clear()
                swaha = 0

        cv2.imshow('Currency Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram terminated by user")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
