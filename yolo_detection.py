"""
Lab 3.1: Pre-trained Object Detection with YOLO
Real-time Person Detection and Tracking

Author: AI Student
Date: 2026
Purpose: Use YOLOv8n to detect people in video frames, count them, and display
         real-time detection results with FPS monitoring.
"""

import os
import cv2
import numpy as np
from collections import deque
from pathlib import Path

# Import YOLO model from ultralytics
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Model configuration
MODEL_NAME = "yolov8n"  # Nano version - fastest for real-time
DETECT_ALL_OBJECTS = True  # Detect all objects, not just persons
TARGET_CLASS = "all"  # Detect all classes

# Video configuration
INPUT_VIDEO_PATH = "input_video/test.mp4"
WEBCAM_ID = 0  # Use 0 for default webcam
USE_WEBCAM = False  # Set to True to use webcam instead of video file

# Output configuration
OUTPUT_VIDEO_DIR = "output_video"
OUTPUT_FRAMES_DIR = "output_frames"
OUTPUT_VIDEO_NAME = "detection_output.mp4"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Display parameters
FONT_THICKNESS = 2
FONT_SCALE = 0.7
BOX_COLOR = (0, 255, 0)  # Green in BGR
TEXT_COLOR = (0, 255, 0)  # Green
TEXT_BG_COLOR = (0, 0, 0)  # Black background

# FPS calculation
FPS_WINDOW = 30  # Calculate FPS over last 30 frames

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_directories():
    for directory in [OUTPUT_VIDEO_DIR, OUTPUT_FRAMES_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")


def load_yolo_model(model_name=MODEL_NAME):
    try:
        print(f"Loading {model_name} model...")
        model = YOLO(f"{model_name}.pt")
        print(f"✓ {model_name} model loaded successfully")
        print(f"  Model device: {model.device}")
        return model
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


def open_video_source(use_webcam=USE_WEBCAM, video_path=INPUT_VIDEO_PATH):
    try:
        if use_webcam:
            cap = cv2.VideoCapture(WEBCAM_ID)
            if not cap.isOpened():
                raise RuntimeError("Cannot open webcam")
            print(f"✓ Webcam opened (ID: {WEBCAM_ID})")
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = None
            
        else:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_path}")
            print(f"✓ Video file opened: {video_path}")
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_info = {
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': frame_count
        }
        
        print(f"  Video info: {width}x{height} @ {fps} FPS")
        if frame_count:
            print(f"  Total frames: {frame_count}")
        
        return cap, video_info
        
    except Exception as e:
        print(f"✗ Error opening video source: {e}")
        raise


def setup_video_writer(output_path, width, height, fps):
    try:
        # Use mp4v codec for better compatibility and playback
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError("Failed to create video writer")
        
        print(f"✓ Video writer created: {output_path}")
        return writer
        
    except Exception as e:
        print(f"✗ Error setting up video writer: {e}")
        raise


def run_yolo_inference(model, frame, conf_threshold=CONFIDENCE_THRESHOLD):
    try:
        # Run inference (returns list of Results objects)
        results = model(frame, conf=conf_threshold, verbose=False)
        return results[0] if results else None
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return None


def filter_detections_by_class(results, detect_all=DETECT_ALL_OBJECTS):
    if results is None or not hasattr(results, 'boxes'):
        return [], [], [], 0
    
    boxes = []
    confidences = []
    class_names = []
    
    try:
        for box in results.boxes:
            # Get class ID and name
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            
            # Get bounding box coordinates (xyxy format) and confidence
            xyxy = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            
            boxes.append(xyxy)
            confidences.append(confidence)
            class_names.append(cls_name)
        
        return boxes, confidences, class_names, len(boxes)
        
    except Exception as e:
        print(f"✗ Error filtering detections: {e}")
        return [], [], 0


def draw_detections(frame, boxes, confidences, class_names, text_scale=0.6):
    frame_copy = frame.copy()
    
    try:
        for (x1, y1, x2, y2), conf, cls_name in zip(boxes, confidences, class_names):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), BOX_COLOR, FONT_THICKNESS)
            
            # Prepare class and confidence text
            conf_text = f"{cls_name} {conf:.2f}"
            
            # Get text size for background
            text_size = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1
            )[0]
            
            # Draw background rectangle for text
            text_x = x1
            text_y = y1 - 5
            cv2.rectangle(
                frame_copy,
                (text_x, text_y - text_size[1] - 4),
                (text_x + text_size[0] + 4, text_y + 4),
                TEXT_BG_COLOR,
                -1
            )
            
            # Draw confidence text
            cv2.putText(
                frame_copy,
                conf_text,
                (text_x + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                TEXT_COLOR,
                1
            )
        
        return frame_copy
        
    except Exception as e:
        print(f"✗ Error drawing detections: {e}")
        return frame


def draw_stats(frame, object_count, fps):
    frame_copy = frame.copy()
    
    try:
        # Object count text
        count_text = f"Objects: {object_count}"
        count_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Draw background for count
        cv2.rectangle(frame_copy, (10, 10), (20 + count_size[0], 40), (0, 0, 0), -1)
        cv2.putText(
            frame_copy,
            count_text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        # FPS text
        fps_text = f"FPS: {fps:.1f}"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Draw background for FPS (top right)
        frame_h = frame_copy.shape[0]
        frame_w = frame_copy.shape[1]
        cv2.rectangle(
            frame_copy,
            (frame_w - 180, 10),
            (frame_w - 10, 40),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            frame_copy,
            fps_text,
            (frame_w - 170, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        return frame_copy
        
    except Exception as e:
        print(f"✗ Error drawing statistics: {e}")
        return frame


class FPSCounter:
    def __init__(self, window_size=FPS_WINDOW):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
    
    def update(self, current_time):
        self.times.append(current_time)
    
    def get_fps(self):
        if len(self.times) < 2:
            return 0.0
        
        time_diff = (self.times[-1] - self.times[0]) / 1000.0  # Convert to seconds
        if time_diff == 0:
            return 0.0
        
        fps = (len(self.times) - 1) / time_diff
        return fps


def save_annotated_frame(frame, frame_num, output_dir=OUTPUT_FRAMES_DIR):
    try:
        filename = os.path.join(output_dir, f"frame_{frame_num:04d}.png")
        cv2.imwrite(filename, frame)
        
    except Exception as e:
        print(f"✗ Error saving frame: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("LAB 3.1: PRE-TRAINED OBJECT DETECTION WITH YOLO")
    print("="*80 + "\n")
    
    try:
        # Step 1: Create output directories
        print("STEP 1: Setting up directories...")
        create_output_directories()
        
        # Step 2: Load YOLO model
        print("\nSTEP 2: Loading YOLO model...")
        model = load_yolo_model()
        
        # Step 3: Open video source
        print("\nSTEP 3: Opening video source...")
        cap, video_info = open_video_source()
        
        # Step 4: Set up output video writer
        print("\nSTEP 4: Setting up video writer...")
        output_path = os.path.join(OUTPUT_VIDEO_DIR, OUTPUT_VIDEO_NAME)
        writer = setup_video_writer(
            output_path,
            video_info['width'],
            video_info['height'],
            video_info['fps']
        )
        
        # Initialize statistics
        fps_counter = FPSCounter()
        frame_count = 0
        total_people = []
        frames_saved = 0
        
        print("\nSTEP 5: Processing frames...")
        print("Press ESC to stop\n")
        
        # Frame processing loop
        while True:
            ret, frame = cap.read()
            
            # Check if frame was read successfully
            if not ret:
                print("\n✓ End of video reached")
                break
            
            frame_count += 1
            
            # Get current timestamp for FPS calculation
            current_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000
            fps_counter.update(current_time)
            
            # Run YOLO inference
            results = run_yolo_inference(model, frame)
            
            # Get all detections
            boxes, confidences, class_names, object_count = filter_detections_by_class(results)
            total_people.append(object_count)
            
            # Draw detections on frame
            annotated_frame = draw_detections(frame, boxes, confidences, class_names)
            
            # Get current FPS
            current_fps = fps_counter.get_fps()
            
            # Draw statistics
            annotated_frame = draw_stats(annotated_frame, object_count, current_fps)
            
            # Write frame to output video
            writer.write(annotated_frame)
            
            # Save sample frames (every 30 frames, up to 5 frames)
            if frame_count % 30 == 0 and frames_saved < 5:
                save_annotated_frame(annotated_frame, frame_count)
                frames_saved += 1
            
            # Display frame (optional - comment out for faster processing)
            cv2.imshow("YOLO Person Detection", annotated_frame)
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count:4d} | Objects: {object_count:2d} | FPS: {current_fps:.1f}")
            
            # Check for ESC key to stop
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("\n✓ Processing stopped by user (ESC pressed)")
                break
        
        # Release resources
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"✓ Total frames processed: {frame_count}")
        print(f"✓ Total frames saved: {frames_saved}")
        print(f"✓ Average objects per frame: {np.mean(total_people):.2f}")
        print(f"✓ Max objects in single frame: {np.max(total_people)}")
        print(f"✓ Average FPS: {fps_counter.get_fps():.1f}")
        print(f"✓ Output video saved: {output_path}")
        print(f"✓ Sample frames saved: {OUTPUT_FRAMES_DIR}/")
        print("✓ Video codec: mp4v (MP4 format for wide compatibility)")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ EXECUTION FAILED: {e}")
        print("="*80 + "\n")
        raise
    finally:
        # Ensure resources are released
        try:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
        except:
            pass


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
