import cv2
import os
import time
from datetime import datetime

# Configuration
SAMPLES_WITH_PILOT = 100
SAMPLES_NO_PILOT = 100
DATA_DIR = '../pilot_presence_dataset'
COUNTDOWN = 3

def create_directories():
    os.makedirs(os.path.join(DATA_DIR, 'pilot_present'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'no_pilot'), exist_ok=True)
    print(f"Created directories in {DATA_DIR}/")

def countdown_display(cap, seconds):
    for i in range(seconds, 0, -1):
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Get ready: {i}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Pilot Presence - Data Collection', frame)
            cv2.waitKey(1000)

def collect_data(label, num_samples):
    print(f"\n{'='*60}")
    print(f"Collecting: {label.upper()}")
    print(f"{'='*60}")
    print(f"Samples needed: {num_samples}")
    
    if label == 'pilot_present':
        print("\nInstructions - WITH PILOT:")
        print("  - Sit in front of camera")
        print("  - Move your head around:")
        print("    * Turn left/right")
        print("    * Tilt up/down")
        print("    * Move closer/farther")
        print("  - Different expressions")
        print("  - Vary lighting (move around room)")
    else:
        print("\nInstructions - NO PILOT (EMPTY STATION):")
        print("  - Point camera at empty chair/desk")
        print("  - Move camera around room")
        print("  - Different angles of empty space")
        print("  - Different rooms/backgrounds")
        print("  - NO PEOPLE should be visible!")
        print("  - Walls, furniture, empty seats OK")
    
    print("\nPress 's' to start, 'p' to pause, 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return False
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    capturing = False
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces for verification
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw face boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # UI
        if capturing:
            color = (0, 255, 0)
            status = "CAPTURING"
        else:
            color = (0, 0, 255)
            status = "READY - Press 's' to start"
        
        # Progress bar
        progress = int((count / num_samples) * 100)
        cv2.rectangle(frame, (10, 10), (630, 40), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (10 + int(620 * progress / 100), 40), color, -1)
        cv2.putText(frame, f"{count}/{num_samples} ({progress}%)", 
                   (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, status, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Label: {label.upper()}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Face detection status
        if label == 'pilot_present':
            if len(faces) > 0:
                cv2.putText(frame, f"Face detected",
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face",
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if len(faces) == 0:
                cv2.putText(frame, f"No face - Good!",
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Face detected - Point at empty space",
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Pilot Presence - Data Collection', frame)
        
        # Capture if active
        if capturing:
            # Verify correct label
            should_have_face = (label == 'pilot_present')
            has_face = len(faces) > 0
            
            if should_have_face == has_face:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{label}_{timestamp}_{count:04d}.jpg"
                filepath = os.path.join(DATA_DIR, label, filename)
                cv2.imwrite(filepath, frame)
                count += 1
                time.sleep(0.05)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if not capturing:
                countdown_display(cap, COUNTDOWN)
                capturing = True
                print("Started capturing...")
        elif key == ord('p'):
            capturing = False
            print("Paused.")
        elif key == ord('q'):
            print("Quitting...")
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ Collected {count} samples for {label}")
    return True

def main():
    print("="*60)
    print("PILOT PRESENCE DETECTION - DATA COLLECTION")
    print("="*60)
    input("\nPress ENTER to start...")
    
    create_directories()
    
    # Collect pilot present
    print("\n" + "="*60)
    print("STEP 1: PILOT PRESENT")
    print("="*60)
    success = collect_data('pilot_present', SAMPLES_WITH_PILOT)
    if not success:
        return
    
    print("\n" + "="*60)
    print("STEP 2: NO PILOT")
    print("="*60)
    input("\nPress ENTER when ready...")
    
    # Collect no pilot
    collect_data('no_pilot', SAMPLES_NO_PILOT)
    
    # Summary
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)
    
    present_count = len([f for f in os.listdir(os.path.join(DATA_DIR, 'pilot_present')) if f.endswith('.jpg')])
    absent_count = len([f for f in os.listdir(os.path.join(DATA_DIR, 'no_pilot')) if f.endswith('.jpg')])
    
    print(f"\nPilot present (faces): {present_count} images")
    print(f"No pilot (empty): {absent_count} images")
    print(f"Total: {present_count + absent_count} images")
    print(f"\nDataset saved in: {DATA_DIR}/")

if __name__ == "__main__":
    main()
