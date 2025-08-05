import json
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse
from scipy import signal
from scipy.ndimage import uniform_filter1d

from pydantic import TypeAdapter
from yolo_pose.schemas.core import FramesData, KeypointLabel


def extract_simple_periodic_pattern(x_pos: List[int], y_pos: List[int], frames: List[int]) -> dict:
    """
    Simple fallback pattern detection using peak/valley detection.
    """
    print("  Using fallback simple pattern detection")
    
    x_array = np.array(x_pos, dtype=float)
    y_array = np.array(y_pos, dtype=float)
    frames_array = np.array(frames)
    
    # Smooth the signal for better peak detection
    from scipy.ndimage import gaussian_filter1d
    y_smooth = gaussian_filter1d(y_array, sigma=2)
    
    # Find peaks and valleys in Y trajectory (typical cycling motion)
    min_cycle_length = 10
    peaks, _ = signal.find_peaks(y_smooth, distance=min_cycle_length, prominence=np.std(y_smooth) * 0.3)
    valleys, _ = signal.find_peaks(-y_smooth, distance=min_cycle_length, prominence=np.std(y_smooth) * 0.3)
    
    # Create cycles from valley to valley (complete pedal stroke)
    if len(valleys) < 2:
        return {'cycles': [], 'pattern_found': False}
    
    cycle_data = []
    for i in range(len(valleys) - 1):
        start_idx = valleys[i]
        end_idx = valleys[i + 1]
        
        if end_idx - start_idx >= min_cycle_length:
            cycle_x = x_array[start_idx:end_idx]
            cycle_y = y_array[start_idx:end_idx]
            cycle_frames = frames_array[start_idx:end_idx]
            
            cycle_data.append({
                'x': cycle_x,
                'y': cycle_y,
                'frames': cycle_frames,
                'length': len(cycle_x),
                'start_frame': frames_array[start_idx],
                'end_frame': frames_array[end_idx]
            })
    
    if len(cycle_data) < 2:
        return {'cycles': [], 'pattern_found': False}
    
    # Calculate average cycle length for normalization
    cycle_lengths = [c['length'] for c in cycle_data]
    target_length = int(np.median(cycle_lengths))
    
    # Normalize all cycles to same length
    normalized_cycles_x = []
    normalized_cycles_y = []
    
    for cycle in cycle_data:
        if len(cycle['x']) >= 5:  # Only use cycles with enough points
            old_indices = np.linspace(0, 1, len(cycle['x']))
            new_indices = np.linspace(0, 1, target_length)
            
            # Linear interpolation for simplicity
            interp_x = np.interp(new_indices, old_indices, cycle['x'])
            interp_y = np.interp(new_indices, old_indices, cycle['y'])
            
            normalized_cycles_x.append(interp_x)
            normalized_cycles_y.append(interp_y)
    
    if not normalized_cycles_x:
        return {'cycles': [], 'pattern_found': False}
    
    # Calculate average pattern and statistics
    avg_pattern_x = np.mean(normalized_cycles_x, axis=0)
    avg_pattern_y = np.mean(normalized_cycles_y, axis=0)
    std_pattern_x = np.std(normalized_cycles_x, axis=0)
    std_pattern_y = np.std(normalized_cycles_y, axis=0)
    
    # Calculate pattern quality metrics
    pattern_consistency = 1.0 / (1.0 + np.mean(std_pattern_x) + np.mean(std_pattern_y))
    
    # Calculate how well each cycle matches the average
    cycle_similarities = []
    for i, (cycle_x, cycle_y) in enumerate(zip(normalized_cycles_x, normalized_cycles_y)):
        # Calculate correlation with average pattern
        corr_x = np.corrcoef(cycle_x, avg_pattern_x)[0, 1] if len(cycle_x) > 1 else 0
        corr_y = np.corrcoef(cycle_y, avg_pattern_y)[0, 1] if len(cycle_y) > 1 else 0
        similarity = (corr_x + corr_y) / 2
        cycle_similarities.append(similarity)
    
    # Cycle timing statistics
    frame_cycles = [(c['start_frame'], c['end_frame']) for c in cycle_data]
    cycle_durations = [c['end_frame'] - c['start_frame'] for c in cycle_data]
    
    print(f"  Simple method found {len(cycle_data)} cycles")
    
    return {
        'pattern_found': True,
        'cycles': frame_cycles,
        'num_cycles': len(frame_cycles),
        'avg_pattern_x': avg_pattern_x.tolist(),
        'avg_pattern_y': avg_pattern_y.tolist(),
        'std_pattern_x': std_pattern_x.tolist(),
        'std_pattern_y': std_pattern_y.tolist(),
        'pattern_consistency': pattern_consistency,
        'cycle_similarities': cycle_similarities,
        'avg_similarity': np.mean(cycle_similarities),
        'cycle_lengths': cycle_durations,
        'avg_cycle_length': np.mean(cycle_durations),
        'cycle_length_std': np.std(cycle_durations),
        'target_length': target_length,
        'estimated_period': int(np.mean(cycle_durations)),
        'all_cycle_data': cycle_data
    }


def extract_periodic_pattern(x_pos: List[int], y_pos: List[int], frames: List[int]) -> dict:
    """
    Extract the actual periodic pattern using cross-correlation and template matching.
    More robust approach for finding true periodic patterns.
    """
    if len(x_pos) < 40:  # Need more data for robust analysis
        return {'cycles': [], 'pattern_found': False}
    
    x_array = np.array(x_pos, dtype=float)
    y_array = np.array(y_pos, dtype=float)
    frames_array = np.array(frames)
    
    print(f"  Analyzing trajectory with {len(x_pos)} points")
    
    # Smooth the signal
    from scipy.ndimage import gaussian_filter1d
    x_smooth = gaussian_filter1d(x_array, sigma=2)
    y_smooth = gaussian_filter1d(y_array, sigma=2)
    complex_smooth = x_smooth + 1j * y_smooth
    
    # Method 1: Autocorrelation to find period
    def find_period_autocorr(input_signal, min_period=10, max_period=None):
        if max_period is None:
            max_period = len(input_signal) // 3
        
        # Calculate autocorrelation
        autocorr = np.correlate(input_signal, input_signal, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks in autocorrelation (excluding the zero-lag peak)
        try:
            peaks, _ = signal.find_peaks(autocorr[min_period:max_period], 
                                       height=np.max(autocorr) * 0.2)  # Lower threshold
            
            if len(peaks) > 0:
                return peaks[0] + min_period  # Add back the offset
        except Exception as e:
            print(f"    Autocorrelation failed: {e}")
        return None
    
    # Find period using different signals
    period_y = find_period_autocorr(y_smooth)
    period_x = find_period_autocorr(x_smooth)
    period_complex = find_period_autocorr(np.abs(complex_smooth))
    
    # Choose the most reliable period estimate
    periods = [p for p in [period_y, period_x, period_complex] if p is not None]
    print(f"  Candidate periods: {periods}")
    
    if not periods:
        # Fallback to simple peak detection if autocorrelation fails
        print("  Autocorrelation failed, using simple peak detection")
        return extract_simple_periodic_pattern(x_pos, y_pos, frames)
    
    # Use median of different estimates
    estimated_period = int(np.median(periods))
    
    print(f"  Estimated period: {estimated_period} frames")
    
    # Method 2: Template matching to find actual cycles (more permissive)
    def extract_cycles_template_matching(input_signal, period):
        cycles = []
        
        # Try different starting points to find the best template
        best_template = None
        best_score = -1
        
        step_size = max(1, period // 20)  # More starting points
        for start_offset in range(0, min(period, len(input_signal) - period), step_size):
            if start_offset + period >= len(input_signal):
                break
                
            template = input_signal[start_offset:start_offset + period]
            
            # Cross-correlate template with entire signal
            correlation = np.correlate(input_signal, template, mode='valid')
            
            # Normalize correlation
            template_norm = np.linalg.norm(template)
            if template_norm == 0:
                continue
                
            signal_norms = np.array([np.linalg.norm(input_signal[i:i+len(template)]) 
                                   for i in range(len(correlation))])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_corr = correlation / (template_norm * signal_norms)
                normalized_corr = np.nan_to_num(normalized_corr)
            
            # Find score of this template
            score = np.mean(normalized_corr)
            
            if score > best_score:
                best_score = score
                best_template = template
        
        if best_template is None:
            return [], None
        
        print(f"    Best template score: {best_score:.3f}")
        
        # Now find all occurrences of the best template (more permissive)
        correlation = np.correlate(input_signal, best_template, mode='valid')
        template_norm = np.linalg.norm(best_template)
        
        cycle_starts = []
        i = 0
        threshold = 0.5  # Lower threshold for cycle detection
        
        while i < len(correlation) - len(best_template):
            # Calculate normalized correlation at this position
            window = input_signal[i:i+len(best_template)]
            if len(window) == len(best_template):
                window_norm = np.linalg.norm(window)
                if window_norm > 0:
                    corr_val = np.dot(best_template, window) / (template_norm * window_norm)
                    
                    # If correlation is high enough, this is a cycle
                    if corr_val > threshold:
                        cycle_starts.append(i)
                        # Skip ahead to avoid overlapping detections
                        i += max(1, len(best_template) // 3)  # Allow more overlap
                    else:
                        i += 1
                else:
                    i += 1
            else:
                break
        
        print(f"    Found {len(cycle_starts)} cycle starts")
        
        # Extract cycles based on detected starts
        cycles = []
        for j in range(len(cycle_starts) - 1):
            start_idx = cycle_starts[j]
            end_idx = cycle_starts[j + 1]
            
            if end_idx - start_idx > len(best_template) // 3:  # More permissive length
                cycles.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'length': end_idx - start_idx
                })
        
        return cycles, best_template
    
    # Extract cycles using template matching
    cycles_y, template_y = extract_cycles_template_matching(y_smooth, estimated_period)
    cycles_x, template_x = extract_cycles_template_matching(x_smooth, estimated_period)
    
    # Use the method that found more cycles
    if len(cycles_y) >= len(cycles_x):
        cycles_info = cycles_y
        print(f"  Using Y-based cycle detection: {len(cycles_y)} cycles")
    else:
        cycles_info = cycles_x
        print(f"  Using X-based cycle detection: {len(cycles_x)} cycles")
    
    if len(cycles_info) < 2:
        return {'cycles': [], 'pattern_found': False}
    
    # Extract cycle data
    cycle_data = []
    for cycle in cycles_info:
        start_idx = cycle['start_idx']
        end_idx = cycle['end_idx']
        
        cycle_x = x_array[start_idx:end_idx]
        cycle_y = y_array[start_idx:end_idx]
        cycle_frames = frames_array[start_idx:end_idx]
        
        cycle_data.append({
            'x': cycle_x,
            'y': cycle_y,
            'frames': cycle_frames,
            'length': len(cycle_x),
            'start_frame': frames_array[start_idx],
            'end_frame': frames_array[end_idx]
        })
    
    # Calculate average cycle length for normalization
    cycle_lengths = [c['length'] for c in cycle_data]
    target_length = int(np.median(cycle_lengths))
    
    # Normalize all cycles to same length using better interpolation
    normalized_cycles_x = []
    normalized_cycles_y = []
    
    for cycle in cycle_data:
        if len(cycle['x']) >= 5:  # Only use cycles with enough points
            # Use spline interpolation for smoother results
            from scipy.interpolate import interp1d
            
            old_indices = np.linspace(0, 1, len(cycle['x']))
            new_indices = np.linspace(0, 1, target_length)
            
            # Create interpolation functions
            try:
                f_x = interp1d(old_indices, cycle['x'], kind='cubic', 
                              bounds_error=False, fill_value='extrapolate')
                f_y = interp1d(old_indices, cycle['y'], kind='cubic', 
                              bounds_error=False, fill_value='extrapolate')
                
                interp_x = f_x(new_indices)
                interp_y = f_y(new_indices)
                
                normalized_cycles_x.append(interp_x)
                normalized_cycles_y.append(interp_y)
            except Exception:
                # Fallback to linear interpolation
                interp_x = np.interp(new_indices, old_indices, cycle['x'])
                interp_y = np.interp(new_indices, old_indices, cycle['y'])
                
                normalized_cycles_x.append(interp_x)
                normalized_cycles_y.append(interp_y)
    
    if not normalized_cycles_x:
        return {'cycles': [], 'pattern_found': False}
    
    # Calculate average pattern and statistics
    avg_pattern_x = np.mean(normalized_cycles_x, axis=0)
    avg_pattern_y = np.mean(normalized_cycles_y, axis=0)
    std_pattern_x = np.std(normalized_cycles_x, axis=0)
    std_pattern_y = np.std(normalized_cycles_y, axis=0)
    
    # Calculate pattern quality metrics
    pattern_consistency = 1.0 / (1.0 + np.mean(std_pattern_x) + np.mean(std_pattern_y))
    
    # Calculate how well each cycle matches the average
    cycle_similarities = []
    for i, (cycle_x, cycle_y) in enumerate(zip(normalized_cycles_x, normalized_cycles_y)):
        # Calculate correlation with average pattern
        corr_x = np.corrcoef(cycle_x, avg_pattern_x)[0, 1]
        corr_y = np.corrcoef(cycle_y, avg_pattern_y)[0, 1]
        similarity = (corr_x + corr_y) / 2
        cycle_similarities.append(similarity)
    
    # Cycle timing statistics
    frame_cycles = [(c['start_frame'], c['end_frame']) for c in cycle_data]
    cycle_durations = [c['end_frame'] - c['start_frame'] for c in cycle_data]
    
    return {
        'pattern_found': True,
        'cycles': frame_cycles,
        'num_cycles': len(frame_cycles),
        'avg_pattern_x': avg_pattern_x.tolist(),
        'avg_pattern_y': avg_pattern_y.tolist(),
        'std_pattern_x': std_pattern_x.tolist(),
        'std_pattern_y': std_pattern_y.tolist(),
        'pattern_consistency': pattern_consistency,
        'cycle_similarities': cycle_similarities,
        'avg_similarity': np.mean(cycle_similarities),
        'cycle_lengths': cycle_durations,
        'avg_cycle_length': np.mean(cycle_durations),
        'cycle_length_std': np.std(cycle_durations),
        'target_length': target_length,
        'estimated_period': estimated_period,
        'all_cycle_data': cycle_data
    }


def analyze_periodic_patterns(frames_data: FramesData, output_dir: Path) -> dict:
    """Analyze periodic patterns in ankle and knee motion without geometric assumptions."""
    
    # Extract data for ankle and knee
    ankle_frames, ankle_x, ankle_y = extract_keypoint_positions(frames_data, KeypointLabel.RIGHT_ANKLE)
    knee_frames, knee_x, knee_y = extract_keypoint_positions(frames_data, KeypointLabel.RIGHT_KNEE)
    
    results = {}
    
    print("\n" + "="*50)
    print("PERIODIC PATTERN ANALYSIS")
    print("="*50)
    
    # Initialize pattern results with default empty values
    ankle_pattern = {'pattern_found': False}
    knee_pattern = {'pattern_found': False}
    
    # Analyze ankle pattern
    if ankle_frames:
        ankle_pattern = extract_periodic_pattern(ankle_x, ankle_y, ankle_frames)
        results['ankle'] = ankle_pattern
        
        if ankle_pattern['pattern_found']:
            print("\nANKLE PERIODIC MOTION:")
            print(f"  Detected cycles: {ankle_pattern['num_cycles']}")
            print(f"  Estimated period: {ankle_pattern['estimated_period']} frames")
            print(f"  Average cycle length: {ankle_pattern['avg_cycle_length']:.1f} frames")
            print(f"  Cycle length std: {ankle_pattern['cycle_length_std']:.1f} frames")
            print(f"  Pattern consistency: {ankle_pattern['pattern_consistency']:.3f}")
            print(f"  Average similarity: {ankle_pattern['avg_similarity']:.3f}")
            print(f"  Pattern length: {ankle_pattern['target_length']} points")
            
            # Calculate motion ranges for the average pattern
            x_range = np.max(ankle_pattern['avg_pattern_x']) - np.min(ankle_pattern['avg_pattern_x'])
            y_range = np.max(ankle_pattern['avg_pattern_y']) - np.min(ankle_pattern['avg_pattern_y'])
            print(f"  Pattern X range: {x_range:.1f} pixels")
            print(f"  Pattern Y range: {y_range:.1f} pixels")
        else:
            print("\nANKLE: No periodic pattern detected")
    
    # Analyze knee pattern
    if knee_frames:
        knee_pattern = extract_periodic_pattern(knee_x, knee_y, knee_frames)
        results['knee'] = knee_pattern
        
        if knee_pattern['pattern_found']:
            print("\nKNEE PERIODIC MOTION:")
            print(f"  Detected cycles: {knee_pattern['num_cycles']}")
            print(f"  Estimated period: {knee_pattern['estimated_period']} frames")
            print(f"  Average cycle length: {knee_pattern['avg_cycle_length']:.1f} frames")
            print(f"  Cycle length std: {knee_pattern['cycle_length_std']:.1f} frames")
            print(f"  Pattern consistency: {knee_pattern['pattern_consistency']:.3f}")
            print(f"  Average similarity: {knee_pattern['avg_similarity']:.3f}")
            print(f"  Pattern length: {knee_pattern['target_length']} points")
            
            # Calculate motion ranges for the average pattern
            x_range = np.max(knee_pattern['avg_pattern_x']) - np.min(knee_pattern['avg_pattern_x'])
            y_range = np.max(knee_pattern['avg_pattern_y']) - np.min(knee_pattern['avg_pattern_y'])
            print(f"  Pattern X range: {x_range:.1f} pixels")
            print(f"  Pattern Y range: {y_range:.1f} pixels")
        else:
            print("\nKNEE: No periodic pattern detected")
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    
    # Plot ankle analysis
    if ankle_frames and ankle_pattern['pattern_found']:
        # Original trajectory
        axes[0, 0].plot(ankle_x, ankle_y, 'lightblue', alpha=0.7, label='Full trajectory')
        axes[0, 0].plot(ankle_pattern['avg_pattern_x'], ankle_pattern['avg_pattern_y'], 
                       'blue', linewidth=3, label='Average pattern')
        axes[0, 0].scatter(ankle_pattern['avg_pattern_x'][0], ankle_pattern['avg_pattern_y'][0], 
                          color='green', s=100, marker='o', label='Pattern start')
        axes[0, 0].set_title('Ankle: Trajectory + Average Pattern')
        axes[0, 0].set_xlabel('X Position (pixels)')
        axes[0, 0].set_ylabel('Y Position (pixels)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_yaxis()
        
        # Individual cycles overlay
        axes[0, 1].plot(ankle_pattern['avg_pattern_x'], ankle_pattern['avg_pattern_y'], 
                       'black', linewidth=3, label='Average pattern')
        for i, cycle in enumerate(ankle_pattern['all_cycle_data'][:5]):  # Show first 5 cycles
            axes[0, 1].plot(cycle['x'], cycle['y'], alpha=0.5, label=f'Cycle {i+1}')
        axes[0, 1].set_title('Ankle: Individual Cycles vs Average')
        axes[0, 1].set_xlabel('X Position (pixels)')
        axes[0, 1].set_ylabel('Y Position (pixels)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_yaxis()
        
        # Pattern consistency (error bars)
        pattern_points = np.arange(len(ankle_pattern['avg_pattern_x']))
        axes[0, 2].errorbar(pattern_points, ankle_pattern['avg_pattern_x'], 
                           yerr=ankle_pattern['std_pattern_x'], 
                           alpha=0.7, label='X pattern ± std')
        axes[0, 2].errorbar(pattern_points, ankle_pattern['avg_pattern_y'], 
                           yerr=ankle_pattern['std_pattern_y'], 
                           alpha=0.7, label='Y pattern ± std')
        axes[0, 2].set_title('Ankle: Pattern Variability')
        axes[0, 2].set_xlabel('Pattern Point Index')
        axes[0, 2].set_ylabel('Position (pixels)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Cycle similarity scores
        cycle_nums = range(1, len(ankle_pattern['cycle_similarities']) + 1)
        axes[0, 3].bar(cycle_nums, ankle_pattern['cycle_similarities'], alpha=0.7)
        axes[0, 3].axhline(y=ankle_pattern['avg_similarity'], color='red', linestyle='--', 
                          label=f'Average: {ankle_pattern["avg_similarity"]:.3f}')
        axes[0, 3].set_title('Ankle: Cycle Similarity Scores')
        axes[0, 3].set_xlabel('Cycle Number')
        axes[0, 3].set_ylabel('Similarity to Average')
        axes[0, 3].legend()
        axes[0, 3].grid(True, alpha=0.3)
    elif ankle_frames:
        # Fallback: show raw trajectory if pattern detection failed
        axes[0, 0].plot(ankle_x, ankle_y, 'lightblue', alpha=0.7, label='Raw trajectory')
        axes[0, 0].scatter(ankle_x[0], ankle_y[0], color='green', s=100, marker='o', label='Start')
        axes[0, 0].scatter(ankle_x[-1], ankle_y[-1], color='red', s=100, marker='s', label='End')
        axes[0, 0].set_title('Ankle: Raw Trajectory (No Pattern Detected)')
        axes[0, 0].set_xlabel('X Position (pixels)')
        axes[0, 0].set_ylabel('Y Position (pixels)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_yaxis()
        
        # Show message in other plots
        for j in range(1, 4):
            axes[0, j].text(0.5, 0.5, 'No periodic pattern\ndetected for ankle', 
                           ha='center', va='center', transform=axes[0, j].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[0, j].set_title(f'Ankle: Pattern Analysis {j+1}')
    else:
        # No ankle data at all
        for j in range(4):
            axes[0, j].text(0.5, 0.5, 'No ankle data\navailable', 
                           ha='center', va='center', transform=axes[0, j].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            axes[0, j].set_title(f'Ankle: No Data {j+1}')
    
    # Plot knee analysis
    if knee_frames and knee_pattern['pattern_found']:
        # Original trajectory
        axes[1, 0].plot(knee_x, knee_y, 'lightcoral', alpha=0.7, label='Full trajectory')
        axes[1, 0].plot(knee_pattern['avg_pattern_x'], knee_pattern['avg_pattern_y'], 
                       'red', linewidth=3, label='Average pattern')
        axes[1, 0].scatter(knee_pattern['avg_pattern_x'][0], knee_pattern['avg_pattern_y'][0], 
                          color='green', s=100, marker='o', label='Pattern start')
        axes[1, 0].set_title('Knee: Trajectory + Average Pattern')
        axes[1, 0].set_xlabel('X Position (pixels)')
        axes[1, 0].set_ylabel('Y Position (pixels)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].invert_yaxis()
        
        # Individual cycles overlay
        axes[1, 1].plot(knee_pattern['avg_pattern_x'], knee_pattern['avg_pattern_y'], 
                       'black', linewidth=3, label='Average pattern')
        for i, cycle in enumerate(knee_pattern['all_cycle_data'][:5]):  # Show first 5 cycles
            axes[1, 1].plot(cycle['x'], cycle['y'], alpha=0.5, label=f'Cycle {i+1}')
        axes[1, 1].set_title('Knee: Individual Cycles vs Average')
        axes[1, 1].set_xlabel('X Position (pixels)')
        axes[1, 1].set_ylabel('Y Position (pixels)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].invert_yaxis()
        
        # Pattern consistency (error bars)
        pattern_points = np.arange(len(knee_pattern['avg_pattern_x']))
        axes[1, 2].errorbar(pattern_points, knee_pattern['avg_pattern_x'], 
                           yerr=knee_pattern['std_pattern_x'], 
                           alpha=0.7, label='X pattern ± std')
        axes[1, 2].errorbar(pattern_points, knee_pattern['avg_pattern_y'], 
                           yerr=knee_pattern['std_pattern_y'], 
                           alpha=0.7, label='Y pattern ± std')
        axes[1, 2].set_title('Knee: Pattern Variability')
        axes[1, 2].set_xlabel('Pattern Point Index')
        axes[1, 2].set_ylabel('Position (pixels)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Cycle similarity scores
        cycle_nums = range(1, len(knee_pattern['cycle_similarities']) + 1)
        axes[1, 3].bar(cycle_nums, knee_pattern['cycle_similarities'], alpha=0.7)
        axes[1, 3].axhline(y=knee_pattern['avg_similarity'], color='red', linestyle='--', 
                          label=f'Average: {knee_pattern["avg_similarity"]:.3f}')
        axes[1, 3].set_title('Knee: Cycle Similarity Scores')
        axes[1, 3].set_xlabel('Cycle Number')
        axes[1, 3].set_ylabel('Similarity to Average')
        axes[1, 3].legend()
        axes[1, 3].grid(True, alpha=0.3)
    elif knee_frames:
        # Fallback: show raw trajectory if pattern detection failed
        axes[1, 0].plot(knee_x, knee_y, 'lightcoral', alpha=0.7, label='Raw trajectory')
        axes[1, 0].scatter(knee_x[0], knee_y[0], color='green', s=100, marker='o', label='Start')
        axes[1, 0].scatter(knee_x[-1], knee_y[-1], color='red', s=100, marker='s', label='End')
        axes[1, 0].set_title('Knee: Raw Trajectory (No Pattern Detected)')
        axes[1, 0].set_xlabel('X Position (pixels)')
        axes[1, 0].set_ylabel('Y Position (pixels)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].invert_yaxis()
        
        # Show message in other plots
        for j in range(1, 4):
            axes[1, j].text(0.5, 0.5, 'No periodic pattern\ndetected for knee', 
                           ha='center', va='center', transform=axes[1, j].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, j].set_title(f'Knee: Pattern Analysis {j+1}')
    else:
        # No knee data at all
        for j in range(4):
            axes[1, j].text(0.5, 0.5, 'No knee data\navailable', 
                           ha='center', va='center', transform=axes[1, j].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            axes[1, j].set_title(f'Knee: No Data {j+1}')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'periodic_patterns_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


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


def save_analysis_results(frames_data: FramesData, output_dir: Path, pattern_results: dict = None) -> None:
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
        
        if pattern_results:
            f.write("PERIODIC PATTERN ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write("Extracted actual motion patterns without geometric assumptions:\n\n")
            
            if 'ankle' in pattern_results and pattern_results['ankle']['pattern_found']:
                ankle = pattern_results['ankle']
                f.write("ANKLE PERIODIC MOTION:\n")
                f.write(f"  Detected cycles: {ankle['num_cycles']}\n")
                f.write(f"  Average cycle length: {ankle['avg_cycle_length']:.1f} frames\n")
                f.write(f"  Cycle length std: {ankle['cycle_length_std']:.1f} frames\n")
                f.write(f"  Pattern consistency: {ankle['pattern_consistency']:.3f}\n")
                f.write(f"  Pattern length: {ankle['target_length']} points\n")
                
                x_range = np.max(ankle['avg_pattern_x']) - np.min(ankle['avg_pattern_x'])
                y_range = np.max(ankle['avg_pattern_y']) - np.min(ankle['avg_pattern_y'])
                f.write(f"  Pattern X range: {x_range:.1f} pixels\n")
                f.write(f"  Pattern Y range: {y_range:.1f} pixels\n\n")
            
            if 'knee' in pattern_results and pattern_results['knee']['pattern_found']:
                knee = pattern_results['knee']
                f.write("KNEE PERIODIC MOTION:\n")
                f.write(f"  Detected cycles: {knee['num_cycles']}\n")
                f.write(f"  Average cycle length: {knee['avg_cycle_length']:.1f} frames\n")
                f.write(f"  Cycle length std: {knee['cycle_length_std']:.1f} frames\n")
                f.write(f"  Pattern consistency: {knee['pattern_consistency']:.3f}\n")
                f.write(f"  Pattern length: {knee['target_length']} points\n")
                
                x_range = np.max(knee['avg_pattern_x']) - np.min(knee['avg_pattern_x'])
                y_range = np.max(knee['avg_pattern_y']) - np.min(knee['avg_pattern_y'])
                f.write(f"  Pattern X range: {x_range:.1f} pixels\n")
                f.write(f"  Pattern Y range: {y_range:.1f} pixels\n\n")
        
        f.write("CYCLING MOTION ANALYSIS\n")
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
        
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 20 + "\n")
        f.write("1. For stable joints (hip, shoulder, elbow, wrist):\n")
        f.write("   - Use average positions for bike fitting measurements\n")
        f.write("   - High variability indicates potential measurement issues\n\n")
        f.write("2. For periodic joints (ankle, knee):\n")
        f.write("   - Use extracted average patterns for analysis\n")
        f.write("   - Pattern consistency indicates measurement quality\n")
        f.write("   - Individual cycles show variation in technique\n")
        f.write("   - Use pattern extremes for range of motion analysis\n\n")
    
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
    
    # Analyze actual periodic patterns (no geometric assumptions)
    print("Analyzing periodic patterns...")
    pattern_results = analyze_periodic_patterns(frames_data, output_dir)
    
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
    save_analysis_results(frames_data, output_dir, pattern_results)
    
    print(f"\nAll visualizations and analysis saved to: {output_dir}")
    print("\nSUMMARY:")
    print("- Use average positions for: hip, shoulder, elbow, wrist (stable joints)")
    print("- Use extracted patterns for: ankle, knee (periodic motion)")
    print("- Check 'comprehensive_analysis.txt' for detailed recommendations")
    
    # Print key pattern findings
    if pattern_results:
        print("\nKey Pattern Findings:")
        if 'ankle' in pattern_results and pattern_results['ankle']['pattern_found']:
            ankle = pattern_results['ankle']
            print(f"- Ankle: {ankle['num_cycles']} cycles, consistency: {ankle['pattern_consistency']:.3f}")
        if 'knee' in pattern_results and pattern_results['knee']['pattern_found']:
            knee = pattern_results['knee']
            print(f"- Knee: {knee['num_cycles']} cycles, consistency: {knee['pattern_consistency']:.3f}")


if __name__ == "__main__":
    main()