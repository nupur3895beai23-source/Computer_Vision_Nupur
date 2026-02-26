import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "yolov8s"  # YOLOv8 Small - better for detecting small objects
WEBCAM_ID = 0  # Default webcam
CONFIDENCE_THRESHOLD = 0.67  # Lower threshold for better small object detection
INPUT_SIZE = 640  # YOLO input size

# Display settings
FONT_THICKNESS = 2
FONT_SCALE = 0.7
BOX_COLOR = (0, 255, 0)  # Green (BGR)
TEXT_COLOR = (255, 255, 255)  # White (BGR)
TEXT_BG_COLOR = (0, 0, 0)  # Black
FPS_WINDOW = 30  # Calculate FPS average over 30 frames

# Model will be loaded and we'll use its native class names
MODEL = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_yolo_model(model_name=MODEL_NAME):
    """Load YOLOv8 model and store globally"""
    global MODEL
    try:
        print(f"Loading {model_name} model...")
        MODEL = YOLO(f"{model_name}.pt")
        print(f"✓ Model loaded successfully")
        print(f"✓ Available classes: {list(MODEL.names.values())}")
        return MODEL
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def get_class_name(class_id):
    """Get class name from YOLO model's native class names"""
    global MODEL
    if MODEL is None:
        return f"Unknown_{class_id}"
    return MODEL.names.get(class_id, f"Unknown_{class_id}")


def draw_detection_box(frame, box, class_id, confidence):
    x1, y1, x2, y2 = map(int, box)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, FONT_THICKNESS)
    
    # Prepare text
    class_name = get_class_name(class_id)
    text = f"{class_name} {confidence:.2f}"
    
    # Get text size for background
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)[0]
    
    # Draw background rectangle for text
    text_x, text_y = x1, y1 - 5
    cv2.rectangle(
        frame,
        (text_x, text_y - text_size[1] - 4),
        (text_x + text_size[0] + 4, text_y + 4),
        TEXT_BG_COLOR,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (text_x + 2, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        TEXT_COLOR,
        1
    )


def draw_stats(frame, total_detections, fps):
    """
    Draw statistics (total detections and FPS) on frame
    
    Args:
        frame: Input frame
        total_detections: Number of objects detected
        fps: Frames per second
    """
    stats_text = f"Objects: {total_detections} | FPS: {fps:.1f}"
    
    # Get text size
    text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE + 0.2, 1)[0]
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (5, 5),
        (5 + text_size[0] + 10, 5 + text_size[1] + 10),
        TEXT_BG_COLOR,
        -1
    )
    
    # Draw statistics text
    cv2.putText(
        frame,
        stats_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE + 0.2,
        (0, 255, 255),  # Cyan
        2
    )


def process_frame(frame, model, conf_threshold=CONFIDENCE_THRESHOLD):
    """
    Process a single frame with YOLO inference
    
    Args:
        frame: Input frame
        model: YOLO model
        conf_threshold: Confidence threshold
        
    Returns:
        Processed frame with detections drawn
        Number of objects detected
        Annotated frame
    """
    try:
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        result = results[0]
        
        detection_count = 0
        
        # Process detections
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                draw_detection_box(frame, [x1, y1, x2, y2], class_id, confidence)
                detection_count += 1
        
        return frame, detection_count
        
    except Exception as e:
        print(f"✗ Error processing frame: {e}")
        return frame, 0


def run_live_camera_feed():
    """
    Main function to run live camera feed with object detection
    """
    global MODEL
    
    # Load model
    model = load_yolo_model()
    MODEL = model
    if model is None:
        return
    
    # Open webcam
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    if not cap.isOpened():
        print(f"✗ Cannot open webcam with ID: {WEBCAM_ID}")
        return
    
    print(f"✓ Webcam opened successfully")
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS tracking
    fps_deque = deque(maxlen=FPS_WINDOW)
    prev_time = time.time()
    
    print("\nStarting live camera feed...")
    print("Press 'q' to quit | 's' to save frame\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Failed to read frame")
                break
            
            # Process frame
            annotated_frame, detection_count = process_frame(frame, model)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            fps_deque.append(fps)
            avg_fps = sum(fps_deque) / len(fps_deque)
            
            # Draw statistics
            draw_stats(annotated_frame, detection_count, avg_fps)
            
            # Display frame
            cv2.imshow("Live Camera Feed - Object Detection", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                print("Exiting...")
                break
            
            elif key == ord('s'):  # Save frame
                filename = f"detection_frame_{frame_count:04d}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"✓ Frame saved: {filename}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Camera feed closed | Processed {frame_count} frames")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_live_camera_feed()
