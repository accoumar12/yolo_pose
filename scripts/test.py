import time
import json
from pathlib import Path
from config import RAW_DATA_DIR_PATH
from ultralytics import YOLO
import cv2

    

def main() -> None:
    # Load your model
    model = YOLO('models/yolo11n-pose.pt')

    keypoint_label_to_index = {
        "nose": 0,
        "right_eye": 2,
        "right_ear": 4,
        "right_shoulder": 6,  
        "right_elbow": 8,
        "right_wrist": 10,
        "right_hip": 12,
        "right_knee": 14,
        "right_ankle": 16,
    }
        

    # Open the video
    input_video_file_name = "test_2.mp4"
    capture = cv2.VideoCapture(RAW_DATA_DIR_PATH / input_video_file_name)
    
    # Get video properties
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    
    # Create experiment directory
    run_id = time.strftime("%Y%m%d-%H%M")
    exp_dir = Path("exp") / f"{run_id}_{input_video_file_name.split('.')[0]}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup output video
    out = cv2.VideoWriter(
        str(exp_dir / "keypoints.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    frames_data = []  # Store data for all frames
    frame_count = 0

    # Process each frame
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # Run model prediction
        results = model(frame)[0]
        
        # Filter keypoints to only show selected ones
        results = results.keypoints.data[0]  # Get keypoints data
        
        # Store frame data
        frame_data = {
            "frame": frame_count,
            "keypoints": {}
        }
        
        # Draw only selected keypoints
        for keypoint_name, idx in keypoint_label_to_index.items():
            if results[idx].sum() > 0:  # Check if keypoint exists
                x, y = results[idx][:2].int().tolist()
                frame_data["keypoints"][keypoint_name] = {"x": x, "y": y}
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw green circle
                cv2.putText(frame, keypoint_name, (x + 10, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        frames_data.append(frame_data)
        frame_count += 1
        
        # Write the frame to output video
        out.write(frame)

    # Save data to JSON file
    json_path = exp_dir / "keypoints.json"
    with open(json_path, 'w') as f:
        json.dump({"frames": frames_data}, f, indent=2)

    # Release everything
    capture.release()
    out.release()

if __name__ == "__main__":
    main()