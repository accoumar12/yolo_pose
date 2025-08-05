import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse
from scipy import signal
from scipy.ndimage import uniform_filter1d

from pydantic import TypeAdapter
from yolo_pose.schemas.core import FramesData, KeypointLabel


def smooth_trajectory(positions: List[int], window_size: int = 5) -> List[float]:
    """Apply moving average smoothing to a trajectory."""
    if len(positions) < window_size:
        return [float(x) for x in positions]
    
    # Convert to numpy array for easier processing
    positions_array = np.array(positions, dtype=float)
    
    # Pad the data to handle edges
    half_window = window_size // 2
    padded = np.pad(positions_array, half_window, mode='edge')
    
    # Apply uniform filter
    smoothed = uniform_filter1d(padded, size=window_size)
    
    # Remove padding to get original length
    result = smoothed[half_window:half_window + len(positions)]
    
    return result.tolist()


def detect_pedaling_cycles(y_positions: List[int], frames: List[int], min_cycle_length: int = 10) -> List[Tuple[int, int]]:
    """
    Detect pedaling cycles based on y-position peaks and valleys.
    Returns list of (start_frame, end_frame) tuples for each cycle.
    """
    if len(y_positions) < min_cycle_length * 2:
        return []
    
    # Smooth the signal for better peak detection
    smoothed_y = uniform_filter1d(np.array(y_positions), size=5)
    
    # Find peaks (top of pedal stroke) and valleys (bottom of pedal stroke)
    peaks, _ = signal.find_peaks(smoothed_y, distance=min_cycle_length, prominence=np.std(smoothed_y) * 0.5)
    valleys, _ = signal.find_peaks(-smoothed_y, distance=min_cycle_length, prominence=np.std(smoothed_y) * 0.5)
    
    cycles = []
    
    # Define cycles from valley to valley (complete pedal stroke)
    for i in range(len(valleys) - 1):
        start_idx = valleys[i]
        end_idx = valleys[i + 1]
        
        if end_idx - start_idx >= min_cycle_length:
            cycles.append((frames[start_idx], frames[end_idx]))
    
    return cycles


def calculate_stable_joint_averages(frames_data: FramesData) -> dict:
    """Calculate average positions for stable joints (hip, shoulder, elbow, wrist)."""
    stable_joints = [KeypointLabel.RIGHT_HIP, KeypointLabel.RIGHT_SHOULDER, 
                    KeypointLabel.RIGHT_ELBOW, KeypointLabel.RIGHT_WRIST]
    
    averages = {}
    
    for joint in stable_joints:
        frames, x_pos, y_pos = extract_keypoint_positions(frames_data, joint)
        
        if x_pos and y_pos:
            averages[joint.value] = {
                'x_avg': np.mean(x_pos),
                'y_avg': np.mean(y_pos),
                'x_std': np.std(x_pos),
                'y_std': np.std(y_pos),
                'total_frames': len(frames)
            }
    
    return averages


def analyze_cycling_motion(frames_data: FramesData, output_dir: Path) -> None:
    """Analyze cycling-specific motion patterns for ankle and knee."""
    cycling_joints = [KeypointLabel.RIGHT_ANKLE, KeypointLabel.RIGHT_KNEE]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, joint in enumerate(cycling_joints):
        frames, x_pos, y_pos = extract_keypoint_positions(frames_data, joint)
        
        if not frames:
            continue
            
        # Smooth the trajectories
        x_smooth = smooth_trajectory(x_pos, window_size=5)
        y_smooth = smooth_trajectory(y_pos, window_size=5)
        
        # Detect cycles based on y-position (vertical motion)
        cycles = detect_pedaling_cycles(y_pos, frames)
        
        # Plot original vs smoothed
        axes[i, 0].plot(frames, y_pos, 'lightblue', alpha=0.7, label='Original')
        axes[i, 0].plot(frames, y_smooth, 'blue', linewidth=2, label='Smoothed')
        
        # Mark detected cycles
        for j, (start_frame, end_frame) in enumerate(cycles):
            axes[i, 0].axvspan(start_frame, end_frame, alpha=0.2, color=f'C{j%10}')
        
        axes[i, 0].set_title(f'{joint.value} - Vertical Motion (Y)')
        axes[i, 0].set_xlabel('Frame')
        axes[i, 0].set_ylabel('Y Position (pixels)')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot 2D trajectory
        axes[i, 1].plot(x_pos, y_pos, 'lightblue', alpha=0.7, label='Original')
        axes[i, 1].plot(x_smooth, y_smooth, 'blue', linewidth=2, label='Smoothed')
        axes[i, 1].scatter(x_pos[0], y_pos[0], color='green', s=100, marker='o', label='Start')
        axes[i, 1].scatter(x_pos[-1], y_pos[-1], color='red', s=100, marker='s', label='End')
        
        axes[i, 1].set_title(f'{joint.value} - 2D Trajectory')
        axes[i, 1].set_xlabel('X Position (pixels)')
        axes[i, 1].set_ylabel('Y Position (pixels)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].invert_yaxis()
        
        # Print cycle information
        print(f"\n{joint.value} Analysis:")
        print(f"  Detected {len(cycles)} pedaling cycles")
        if cycles:
            cycle_lengths = [end - start for start, end in cycles]
            print(f"  Average cycle length: {np.mean(cycle_lengths):.1f} frames")
            print(f"  Cycle length std: {np.std(cycle_lengths):.1f} frames")
        
        # Calculate range of motion
        x_range = max(x_pos) - min(x_pos)
        y_range = max(y_pos) - min(y_pos)
        print(f"  Range of motion: {x_range}px (X), {y_range}px (Y)")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cycling_motion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cycles


def load_keypoints_data(json_path: Path) -> FramesData:
    """Load keypoints data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    adapter = TypeAdapter(FramesData)
    return adapter.validate_python(data)


def extract_keypoint_positions(frames_data: FramesData, keypoint: KeypointLabel) -> tuple[List[int], List[int], List[int]]:
    """Extract x, y positions and frame numbers for a specific keypoint."""
    frames = []
    x_positions = []
    y_positions = []
    
    for frame_data in frames_data:
        if keypoint in frame_data.keypoints:
            frames.append(frame_data.frame)
            x_positions.append(frame_data.keypoints[keypoint].x)
            y_positions.append(frame_data.keypoints[keypoint].y)
    
    return frames, x_positions, y_positions


def plot_keypoint_trajectories(frames_data: FramesData, output_dir: Path) -> None:
    """Plot x and y trajectories for all keypoints over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Define colors for each keypoint
    colors = {
        KeypointLabel.RIGHT_ANKLE: 'red',
        KeypointLabel.RIGHT_KNEE: 'blue',
        KeypointLabel.RIGHT_HIP: 'green',
        KeypointLabel.RIGHT_ELBOW: 'orange',
        KeypointLabel.RIGHT_SHOULDER: 'purple',
        KeypointLabel.RIGHT_WRIST: 'brown'
    }
    
    for keypoint in KeypointLabel:
        frames, x_pos, y_pos = extract_keypoint_positions(frames_data, keypoint)
        
        if frames:  # Only plot if we have data
            color = colors.get(keypoint, 'black')
            
            # Plot X trajectory
            ax1.plot(frames, x_pos, label=keypoint.value, color=color, marker='o', markersize=2)
            
            # Plot Y trajectory (inverted since y=0 is at top in image coordinates)
            ax2.plot(frames, [-y for y in y_pos], label=keypoint.value, color=color, marker='o', markersize=2)
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X Position (pixels)')
    ax1.set_title('Horizontal Movement of Keypoints Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y Position (pixels, inverted)')
    ax2.set_title('Vertical Movement of Keypoints Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'keypoint_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_keypoint_displacement(frames_data: FramesData, output_dir: Path) -> None:
    """Plot displacement magnitude for each keypoint over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        KeypointLabel.RIGHT_ANKLE: 'red',
        KeypointLabel.RIGHT_KNEE: 'blue',
        KeypointLabel.RIGHT_HIP: 'green',
        KeypointLabel.RIGHT_ELBOW: 'orange',
        KeypointLabel.RIGHT_SHOULDER: 'purple',
        KeypointLabel.RIGHT_WRIST: 'brown'
    }
    
    for keypoint in KeypointLabel:
        frames, x_pos, y_pos = extract_keypoint_positions(frames_data, keypoint)
        
        if len(frames) > 1:  # Need at least 2 points to calculate displacement
            displacements = []
            displacement_frames = []
            
            for i in range(1, len(frames)):
                dx = x_pos[i] - x_pos[i-1]
                dy = y_pos[i] - y_pos[i-1]
                displacement = np.sqrt(dx**2 + dy**2)
                displacements.append(displacement)
                displacement_frames.append(frames[i])
            
            color = colors.get(keypoint, 'black')
            ax.plot(displacement_frames, displacements, label=keypoint.value, color=color, marker='o', markersize=2)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Displacement (pixels)')
    ax.set_title('Frame-to-Frame Displacement of Keypoints')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'keypoint_displacements.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_2d_trajectory(frames_data: FramesData, output_dir: Path) -> None:
    """Plot 2D trajectory showing the path of each keypoint in space."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        KeypointLabel.RIGHT_ANKLE: 'red',
        KeypointLabel.RIGHT_KNEE: 'blue',
        KeypointLabel.RIGHT_HIP: 'green',
        KeypointLabel.RIGHT_ELBOW: 'orange',
        KeypointLabel.RIGHT_SHOULDER: 'purple',
        KeypointLabel.RIGHT_WRIST: 'brown'
    }
    
    for keypoint in KeypointLabel:
        frames, x_pos, y_pos = extract_keypoint_positions(frames_data, keypoint)
        
        if frames:  # Only plot if we have data
            color = colors.get(keypoint, 'black')
            
            # Plot trajectory
            ax.plot(x_pos, y_pos, label=keypoint.value, color=color, alpha=0.7)
            
            # Mark start and end points
            if len(x_pos) > 0:
                ax.scatter(x_pos[0], y_pos[0], color=color, s=100, marker='o', edgecolor='black', linewidth=2, label=f'{keypoint.value} start')
                ax.scatter(x_pos[-1], y_pos[-1], color=color, s=100, marker='s', edgecolor='black', linewidth=2, label=f'{keypoint.value} end')
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('2D Trajectory of Keypoints')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    
    plt.tight_layout()
    plt.savefig(output_dir / '2d_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_movement_statistics(frames_data: FramesData, output_dir: Path) -> None:
    """Analyze and print movement statistics for each keypoint."""
    stats = {}
    
    for keypoint in KeypointLabel:
        frames, x_pos, y_pos = extract_keypoint_positions(frames_data, keypoint)
        
        if len(frames) > 1:
            # Calculate total displacement
            total_displacement = 0
            max_displacement = 0
            displacements = []
            
            for i in range(1, len(frames)):
                dx = x_pos[i] - x_pos[i-1]
                dy = y_pos[i] - y_pos[i-1]
                displacement = np.sqrt(dx**2 + dy**2)
                displacements.append(displacement)
                total_displacement += displacement
                max_displacement = max(max_displacement, displacement)
            
            # Calculate range of motion
            x_range = max(x_pos) - min(x_pos) if x_pos else 0
            y_range = max(y_pos) - min(y_pos) if y_pos else 0
            
            stats[keypoint.value] = {
                'total_displacement': total_displacement,
                'average_displacement_per_frame': np.mean(displacements),
                'max_displacement_per_frame': max_displacement,
                'x_range': x_range,
                'y_range': y_range,
                'total_frames': len(frames)
            }
    
    # Save statistics to file
    stats_file = output_dir / 'movement_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("Movement Statistics for Keypoints\n")
        f.write("=" * 40 + "\n\n")
        
        for keypoint, data in stats.items():
            f.write(f"{keypoint.upper()}:\n")
            f.write(f"  Total frames tracked: {data['total_frames']}\n")
            f.write(f"  Total displacement: {data['total_displacement']:.2f} pixels\n")
            f.write(f"  Average displacement per frame: {data['average_displacement_per_frame']:.2f} pixels\n")
            f.write(f"  Maximum displacement per frame: {data['max_displacement_per_frame']:.2f} pixels\n")
            f.write(f"  X range of motion: {data['x_range']} pixels\n")
            f.write(f"  Y range of motion: {data['y_range']} pixels\n")
            f.write("\n")
    
    print(f"Movement statistics saved to: {stats_file}")
    
    # Print summary to console
    print("\nMovement Statistics Summary:")
    print("-" * 40)
    for keypoint, data in stats.items():
        print(f"{keypoint}: {data['total_displacement']:.1f}px total, {data['average_displacement_per_frame']:.1f}px avg/frame")


def save_analysis_results(frames_data: FramesData, output_dir: Path) -> None:
    """Save comprehensive analysis results to a text file."""
    results_file = output_dir / 'comprehensive_analysis.txt'
    
    # Calculate stable joint averages
    stable_averages = calculate_stable_joint_averages(frames_data)
    
    # Analyze cycling joints
    cycling_joints = [KeypointLabel.RIGHT_ANKLE, KeypointLabel.RIGHT_KNEE]
    cycling_analysis = {}
    
    for joint in cycling_joints:
        frames, x_pos, y_pos = extract_keypoint_positions(frames_data, joint)
        if frames:
            cycles = detect_pedaling_cycles(y_pos, frames)
            x_range = max(x_pos) - min(x_pos)
            y_range = max(y_pos) - min(y_pos)
            
            cycling_analysis[joint.value] = {
                'cycles_detected': len(cycles),
                'x_range': x_range,
                'y_range': y_range,
                'total_frames': len(frames)
            }
            
            if cycles:
                cycle_lengths = [end - start for start, end in cycles]
                cycling_analysis[joint.value].update({
                    'avg_cycle_length': np.mean(cycle_lengths),
                    'cycle_length_std': np.std(cycle_lengths)
                })
    
    # Write results to file
    with open(results_file, 'w') as f:
        f.write("BIKE FITTING KEYPOINT ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("STABLE JOINTS ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write("These joints should remain relatively stable during cycling:\n\n")
        
        for joint, data in stable_averages.items():
            f.write(f"{joint.upper()}:\n")
            f.write(f"  Average position: ({data['x_avg']:.1f}, {data['y_avg']:.1f}) pixels\n")
            f.write(f"  Variability (std): ({data['x_std']:.1f}, {data['y_std']:.1f}) pixels\n")
            f.write(f"  Frames tracked: {data['total_frames']}\n")
            f.write(f"  Stability score: {100 / (1 + data['x_std'] + data['y_std']):.1f}/100\n")
            f.write("\n")
        
        f.write("\nCYCLING MOTION ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write("These joints show periodic motion during pedaling:\n\n")
        
        for joint, data in cycling_analysis.items():
            f.write(f"{joint.upper()}:\n")
            f.write(f"  Cycles detected: {data['cycles_detected']}\n")
            f.write(f"  Range of motion: {data['x_range']}px (X), {data['y_range']}px (Y)\n")
            f.write(f"  Total frames: {data['total_frames']}\n")
            
            if 'avg_cycle_length' in data:
                f.write(f"  Average cycle length: {data['avg_cycle_length']:.1f} frames\n")
                f.write(f"  Cycle consistency (std): {data['cycle_length_std']:.1f} frames\n")
            
            f.write("\n")
        
        f.write("\nRECOMMendations:\n")
        f.write("-" * 20 + "\n")
        f.write("1. For stable joints (hip, shoulder, elbow, wrist):\n")
        f.write("   - Use average positions for bike fitting measurements\n")
        f.write("   - High variability indicates potential measurement issues\n\n")
        f.write("2. For cycling joints (ankle, knee):\n")
        f.write("   - Analyze range of motion and cycle patterns\n")
        f.write("   - Consider filtering/smoothing for cleaner measurements\n")
        f.write("   - Use cycle-averaged positions or extreme positions\n\n")
    
    print(f"Comprehensive analysis saved to: {results_file}")


def main():
    """Main function to process keypoints data and create visualizations."""
    parser = argparse.ArgumentParser(description='Post-process keypoints data and create visualizations')
    parser.add_argument('--exp-dir', type=str, help='Experiment directory containing keypoints.json')
    parser.add_argument('--json-file', type=str, help='Direct path to keypoints.json file')
    
    args = parser.parse_args()
    
    # Determine input file
    if args.json_file:
        json_path = Path(args.json_file)
        output_dir = json_path.parent
    elif args.exp_dir:
        exp_dir = Path(args.exp_dir)
        json_path = exp_dir / 'keypoints.json'
        output_dir = exp_dir
    else:
        # Default to latest experiment
        from config import EXP_DIR_PATH
        exp_dirs = [d for d in EXP_DIR_PATH.iterdir() if d.is_dir()]
        if not exp_dirs:
            print("No experiment directories found. Please specify --exp-dir or --json-file")
            return
        
        latest_exp = max(exp_dirs, key=lambda x: x.name)
        json_path = latest_exp / 'keypoints.json'
        output_dir = latest_exp
        print(f"Using latest experiment: {latest_exp.name}")
    
    if not json_path.exists():
        print(f"Keypoints file not found: {json_path}")
        return
    
    print(f"Loading keypoints data from: {json_path}")
    
    # Load data
    frames_data = load_keypoints_data(json_path)
    print(f"Loaded {len(frames_data)} frames of data")
    
    # Calculate stable joint averages
    print("Calculating stable joint averages...")
    stable_averages = calculate_stable_joint_averages(frames_data)
    
    # Print stable joint averages
    print("\nStable Joint Averages:")
    print("-" * 40)
    for joint, data in stable_averages.items():
        print(f"{joint.upper()}:")
        print(f"  Average position: ({data['x_avg']:.1f}, {data['y_avg']:.1f})")
        print(f"  Standard deviation: ({data['x_std']:.1f}, {data['y_std']:.1f})")
        print(f"  Frames tracked: {data['total_frames']}")
        print()
    
    # Analyze cycling motion for ankle and knee
    print("Analyzing cycling motion patterns...")
    analyze_cycling_motion(frames_data, output_dir)
    
    # Create traditional visualizations
    print("Creating trajectory plots...")
    plot_keypoint_trajectories(frames_data, output_dir)
    
    print("Creating displacement plots...")
    plot_keypoint_displacement(frames_data, output_dir)
    
    print("Creating 2D trajectory plot...")
    plot_2d_trajectory(frames_data, output_dir)
    
    print("Analyzing movement statistics...")
    analyze_movement_statistics(frames_data, output_dir)
    
    print("Generating comprehensive analysis report...")
    save_analysis_results(frames_data, output_dir)
    
    print(f"\nAll visualizations and analysis saved to: {output_dir}")
    print("\nSUMMARY:")
    print("- Use average positions for: hip, shoulder, elbow, wrist (stable joints)")
    print("- Apply smoothing/filtering for: ankle, knee (periodic motion)")
    print("- Check 'comprehensive_analysis.txt' for detailed recommendations")


if __name__ == "__main__":
    main()