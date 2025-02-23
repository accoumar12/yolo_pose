import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_keypoints(json_path):
    """Load keypoints from JSON file and convert to DataFrame."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = []
    for frame_idx, frame_data in enumerate(data):
        for detection in frame_data:
            keypoints = detection['keypoints']
            # Reshape keypoints from [x1,y1,conf1,x2,y2,conf2,...] to [[x1,y1,conf1],[x2,y2,conf2],...]
            keypoints_array = np.array(keypoints).reshape(-1, 3)
            
            frame_dict = {
                'frame': frame_idx,
                'confidence': detection['confidence']
            }
            
            # Add each keypoint's coordinates and confidence
            for i, (x, y, conf) in enumerate(keypoints_array):
                frame_dict[f'kp{i}_x'] = x
                frame_dict[f'kp{i}_y'] = y
                frame_dict[f'kp{i}_conf'] = conf
                
            frames.append(frame_dict)
    
    return pd.DataFrame(frames)

def plot_keypoint_trajectory(df, keypoint_idx, output_path=None):
    """Plot the x,y trajectory of a specific keypoint over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df[f'kp{keypoint_idx}_x'], df[f'kp{keypoint_idx}_y'], 'b-', label=f'Keypoint {keypoint_idx}')
    plt.title(f'Trajectory of Keypoint {keypoint_idx}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_keypoint_height(df, keypoint_idx, output_path=None):
    """Plot the height (y-coordinate) of a specific keypoint over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['frame'], df[f'kp{keypoint_idx}_y'], 'r-', label=f'Keypoint {keypoint_idx}')
    plt.title(f'Height of Keypoint {keypoint_idx} Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    if output_path:
        plt.savefig(output_path)
    plt.show()

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def plot_joint_angle(df, kp1, kp2, kp3, output_path=None):
    """Plot the angle between three keypoints over time."""
    angles = []
    for _, row in df.iterrows():
        p1 = (row[f'kp{kp1}_x'], row[f'kp{kp1}_y'])
        p2 = (row[f'kp{kp2}_x'], row[f'kp{kp2}_y'])
        p3 = (row[f'kp{kp3}_x'], row[f'kp{kp3}_y'])
        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['frame'], angles, 'g-', label=f'Angle {kp1}-{kp2}-{kp3}')
    plt.title(f'Joint Angle {kp1}-{kp2}-{kp3} Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    if output_path:
        plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    # Example usage
    json_path = "path/to/your/keypoints.json"
    df = load_keypoints(json_path)
    
    # Example plots
    plot_keypoint_trajectory(df, keypoint_idx=0)  # Plot trajectory of first keypoint
    plot_keypoint_height(df, keypoint_idx=0)  # Plot height of first keypoint
    plot_joint_angle(df, kp1=5, kp2=6, kp3=7)  # Plot angle between keypoints 5,6,7
