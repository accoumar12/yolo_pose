from enum import Enum
import time

from pydantic import BaseModel, TypeAdapter
from config import EXP_DIR_PATH, RAW_DATA_DIR_PATH
from ultralytics import YOLO
import cv2


class KeypointLabel(str, Enum):
    RIGHT_ANKLE = "right_ankle"
    RIGHT_KNEE = "right_knee"
    RIGHT_HIP = "right_hip"
    RIGHT_ELBOW = "right_elbow"
    RIGHT_SHOULDER = "right_shoulder"
    RIGHT_WRIST = "right_wrist"


keypoint_label_to_index = {
    KeypointLabel.RIGHT_ANKLE: 16,
    KeypointLabel.RIGHT_KNEE: 14,
    KeypointLabel.RIGHT_HIP: 12,
    KeypointLabel.RIGHT_ELBOW: 8,
    KeypointLabel.RIGHT_SHOULDER: 6,
    KeypointLabel.RIGHT_WRIST: 10,
}


class Position(BaseModel):
    x: int
    y: int


class FrameData(BaseModel):
    frame: int
    keypoints: dict[KeypointLabel, Position]

FramesData = list[FrameData]

def main() -> None:
    model = YOLO("models/yolo11n-pose.pt")

    # Open the video
    input_video_file_name = "sample_1.mp4"
    capture = cv2.VideoCapture(RAW_DATA_DIR_PATH / input_video_file_name)

    # Get video properties
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    # Create experiment directory
    run_id = time.strftime("%Y%m%d-%H%M")
    exp_dir = EXP_DIR_PATH / f"{run_id}_{input_video_file_name.split('.')[0]}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup output video
    out = cv2.VideoWriter(
        str(exp_dir / "keypoints.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    frames_data: list[FrameData] = []  # Store data for all frames
    frame_count = 0

    # Process each frame
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        if frame_count > 10:
            break

        # Run model prediction
        results = model(frame)[0]

        # Filter keypoints to only show selected ones
        results = results.keypoints.data[0]  # Get keypoints data

        keypoints = {}
        # Draw only selected keypoints
        for keypoint_name, idx in keypoint_label_to_index.items():
            if results[idx].sum() > 0:  # Check if keypoint exists
                x, y = results[idx][:2].int().tolist()
                keypoints[keypoint_name] = {"x": x, "y": y}
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw green circle
                cv2.putText(
                    frame,
                    keypoint_name,
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        frame_data = FrameData(frame=frame_count, keypoints=keypoints)

        frames_data.append(frame_data)
        frame_count += 1

        # Write the frame to output video
        out.write(frame)

    # Save data to JSON file
    json_path = exp_dir / "keypoints.json"
    adapter = TypeAdapter(FramesData)
    with open(json_path, "w") as f:
        f.write(adapter.dump_json(frames_data, indent=2).decode())
        

    # Release everything
    capture.release()
    out.release()


if __name__ == "__main__":
    main()
