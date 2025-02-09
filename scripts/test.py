from ultralytics import YOLO
import cv2

def main() -> None:
    # Load your model
    model = YOLO('models/yolo11n-pose.pt')

    # Open the video
    capture = cv2.VideoCapture("resources/test.mp4")
    
    # Get video properties
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    out = cv2.VideoWriter(
        'output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    # Process each frame
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # Run model prediction
        results = model(frame)
        
        # Plot the results on the frame
        annotated_frame = results[0].plot()
        
        # Write the frame to output video
        out.write(annotated_frame)

    # Release everything
    capture.release()
    out.release()

if __name__ == "__main__":
    main()