import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
import colorsys
import os

class FlowVisualizer:
    def __init__(self, focal_length=910, history_length=150, ground_truth=None):
        """
        Initialize the flow visualizer.
        
        Args:
            focal_length: Camera focal length in pixels
            history_length: Number of frames to keep in history for plots
            ground_truth: Optional ground truth data (Nx2 array of pitch, yaw)
        """
        self.focal_length = focal_length
        self.history_length = history_length
        self.ground_truth = ground_truth
        self.frame_idx = 0
        
        # History buffers for plotting
        self.pitch_history = deque(maxlen=history_length)
        self.yaw_history = deque(maxlen=history_length)
        self.speed_history = deque(maxlen=history_length)
        
        # Ground truth history if available
        if ground_truth is not None:
            self.gt_pitch_history = deque(maxlen=history_length)
            self.gt_yaw_history = deque(maxlen=history_length)
        else:
            self.gt_pitch_history = None
            self.gt_yaw_history = None
        
        # Flow parameters - balanced for accuracy vs noise
        self.flow_params = dict(
            pyr_scale=0.5,      # Pyramid scale
            levels=3,           # Standard pyramid levels
            winsize=15,         # Medium window size - larger = smoother but less accurate
            iterations=3,       # Standard iterations
            poly_n=5,           # Standard polynomial expansion
            poly_sigma=1.2,     # Standard Gaussian smoothing
            flags=0             # Default flags
        )
        
        # Sky exclusion parameters
        self.sky_exclusion_ratio = 0.2  # Exclude top 20% of image
        
        # FOE smoothing parameters
        self.foe_history_x = deque(maxlen=10)  # Keep last 10 FOE positions
        self.foe_history_y = deque(maxlen=10)
        self.smoothing_alpha = 0.3  # Exponential smoothing factor (0-1, lower = more smoothing)
        self.outlier_threshold = 100  # Pixels - reject FOE jumps larger than this
        
        # Alternative: Running average for angle smoothing
        self.pitch_smooth = deque(maxlen=5)  # Keep last 5 pitch values
        self.yaw_smooth = deque(maxlen=5)     # Keep last 5 yaw values
        
        # CLAHE for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Brightness tracking
        self.prev_brightness = None
        self.brightness_change_threshold = 10  # Threshold for detecting significant brightness changes
        
    
    def compute_flow_magnitude_angle(self, flow):
        """Convert flow field to magnitude and angle representation."""
        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        angle = np.arctan2(flow[:,:,1], flow[:,:,0])
        return magnitude, angle
    
    def create_flow_hsv_visualization(self, flow):
        """
        Create HSV color-coded flow visualization.
        Hue represents direction, Saturation represents magnitude.
        """
        magnitude, angle = self.compute_flow_magnitude_angle(flow)
        
        # Create HSV image
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        
        # Hue: direction (0-179 in OpenCV)
        hsv[:,:,0] = ((angle + np.pi) * 180 / (2 * np.pi)).astype(np.uint8)
        
        # Saturation: always max
        hsv[:,:,1] = 255
        
        # Value: normalized magnitude
        normalized_mag = np.clip(magnitude * 10, 0, 255)
        hsv[:,:,2] = normalized_mag.astype(np.uint8)
        
        # Convert to BGR for display
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
    
    def create_flow_arrow_visualization(self, img, flow, foe_x, foe_y, raw_foe_x=None, raw_foe_y=None):
        """Create visualization with flow arrows and FOE."""
        vis = img.copy()
        h, w = img.shape[:2]
        
        # Draw flow vectors
        step = 25  # Grid spacing
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        
        # Scale flow vectors for visibility
        scale = 3
        
        for i in range(len(x)):
            # Skip very small flows
            if np.sqrt(fx[i]**2 + fy[i]**2) < 0.5:
                continue
                
            # Draw arrow
            start_point = (x[i], y[i])
            end_point = (int(x[i] + fx[i]*scale), int(y[i] + fy[i]*scale))
            
            # Color based on magnitude
            magnitude = np.sqrt(fx[i]**2 + fy[i]**2)
            color = self.get_magnitude_color(magnitude)
            
            cv2.arrowedLine(vis, start_point, end_point, color, 2, tipLength=0.3)
        
        # Draw raw FOE position if available (small gray marker)
        if raw_foe_x is not None and raw_foe_y is not None:
            if 0 <= raw_foe_x < w and 0 <= raw_foe_y < h:
                cv2.circle(vis, (int(raw_foe_x), int(raw_foe_y)), 5, (128, 128, 128), 1)
                cv2.putText(vis, "raw", (int(raw_foe_x) + 8, int(raw_foe_y) - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
        
        # Draw smoothed FOE with crosshair
        if 0 <= foe_x < w and 0 <= foe_y < h:
            # Crosshair
            cv2.line(vis, (int(foe_x)-20, int(foe_y)), (int(foe_x)+20, int(foe_y)), (0, 0, 255), 2)
            cv2.line(vis, (int(foe_x), int(foe_y)-20), (int(foe_x), int(foe_y)+20), (0, 0, 255), 2)
            # Circle
            cv2.circle(vis, (int(foe_x), int(foe_y)), 10, (0, 0, 255), 2)
            # Label
            cv2.putText(vis, "FOE", (int(foe_x) + 15, int(foe_y) - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw center reference
        cv2.circle(vis, (w//2, h//2), 5, (255, 255, 0), -1)
        cv2.drawMarker(vis, (w//2, h//2), (255, 255, 0), cv2.MARKER_CROSS, 20, 1)
        
        # Show excluded sky region
        # Draw line showing excluded sky region (top 20%)
        sky_line_y = int(h * self.sky_exclusion_ratio)
        cv2.line(vis, (0, sky_line_y), (w-1, sky_line_y), (128, 128, 255), 1)
        cv2.putText(vis, "Sky exclusion", (10, sky_line_y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 255), 1)
        
        return vis
    
    def get_magnitude_color(self, magnitude, max_magnitude=10):
        """Get color based on flow magnitude (blue to red gradient)."""
        normalized = np.clip(magnitude / max_magnitude, 0, 1)
        # Blue -> Green -> Red gradient
        if normalized < 0.5:
            # Blue to Green
            r = 0
            g = int(255 * (normalized * 2))
            b = int(255 * (1 - normalized * 2))
        else:
            # Green to Red
            r = int(255 * ((normalized - 0.5) * 2))
            g = int(255 * (1 - (normalized - 0.5) * 2))
            b = 0
        return (b, g, r)
    
    def create_motion_trail(self, img, history_length=30):
        """Create visualization showing recent FOE positions as a trail."""
        vis = img.copy()
        
        if len(self.yaw_history) < 2:
            return vis
            
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Get recent history
        recent_yaws = list(self.yaw_history)[-history_length:]
        recent_pitches = list(self.pitch_history)[-history_length:]
        
        # Draw trail
        for i in range(1, len(recent_yaws)):
            # Convert angles back to pixel coordinates
            x1 = cx + self.focal_length * np.tan(recent_yaws[i-1])
            y1 = cy - self.focal_length * np.tan(recent_pitches[i-1])
            x2 = cx + self.focal_length * np.tan(recent_yaws[i])
            y2 = cy - self.focal_length * np.tan(recent_pitches[i])
            
            # Fade effect
            alpha = i / len(recent_yaws)
            color = (int(255 * alpha), int(100 * alpha), int(255 * (1-alpha)))
            thickness = int(1 + alpha * 2)
            
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        return vis
    
    def create_dashboard(self, pitch, yaw, speed_estimate):
        """Create a dashboard showing current metrics."""
        # Create figure
        fig = plt.figure(figsize=(4, 6), facecolor='black')
        fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.3)
        
        # Current values display
        ax1 = plt.subplot(3, 1, 1)
        ax1.set_facecolor('black')
        ax1.axis('off')
        
        # Display current values
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        
        ax1.text(0.5, 0.9, 'Direction', ha='center', fontsize=16, color='white', weight='bold')
        ax1.text(0.5, 0.65, f'Yaw: {yaw_deg:+.1f}°', ha='center', fontsize=14, color='cyan')
        ax1.text(0.5, 0.45, f'Pitch: {pitch_deg:+.1f}°', ha='center', fontsize=14, color='lime')
        
        # Show ground truth if available
        if self.ground_truth is not None and self.frame_idx < len(self.ground_truth):
            gt_pitch_deg = np.degrees(self.ground_truth[self.frame_idx, 0])
            gt_yaw_deg = np.degrees(self.ground_truth[self.frame_idx, 1])
            ax1.text(0.5, 0.2, f'GT Yaw: {gt_yaw_deg:+.1f}°', ha='center', fontsize=12, color='lightblue')
            ax1.text(0.5, 0.05, f'GT Pitch: {gt_pitch_deg:+.1f}°', ha='center', fontsize=12, color='lightgreen')
        
        # Yaw history plot
        ax2 = plt.subplot(3, 1, 2)
        ax2.set_facecolor('#111111')
        if len(self.yaw_history) > 1:
            x = np.arange(len(self.yaw_history))
            y = np.degrees(list(self.yaw_history))
            ax2.plot(x, y, 'cyan', linewidth=2, label='Predicted')
            ax2.fill_between(x, y, alpha=0.3, color='cyan')
        
        # Plot ground truth yaw if available
        if self.gt_yaw_history and len(self.gt_yaw_history) > 1:
            x_gt = np.arange(len(self.gt_yaw_history))
            y_gt = np.degrees(list(self.gt_yaw_history))
            ax2.plot(x_gt, y_gt, 'lightblue', linewidth=2, linestyle='--', label='Ground Truth')
        
        ax2.set_ylabel('Yaw (°)', color='white')
        ax2.set_xlim(0, self.history_length)
        ax2.set_ylim(-30, 30)
        ax2.grid(True, alpha=0.2)
        ax2.tick_params(colors='white')
        if self.gt_yaw_history and len(self.yaw_history) > 1:
            ax2.legend(loc='upper right', fontsize=8)
        
        # Pitch history plot
        ax3 = plt.subplot(3, 1, 3)
        ax3.set_facecolor('#111111')
        if len(self.pitch_history) > 1:
            x = np.arange(len(self.pitch_history))
            y = np.degrees(list(self.pitch_history))
            ax3.plot(x, y, 'lime', linewidth=2, label='Predicted')
            ax3.fill_between(x, y, alpha=0.3, color='lime')
        
        # Plot ground truth pitch if available
        if self.gt_pitch_history and len(self.gt_pitch_history) > 1:
            x_gt = np.arange(len(self.gt_pitch_history))
            y_gt = np.degrees(list(self.gt_pitch_history))
            ax3.plot(x_gt, y_gt, 'lightgreen', linewidth=2, linestyle='--', label='Ground Truth')
        
        ax3.set_ylabel('Pitch (°)', color='white')
        ax3.set_xlabel('Frame', color='white')
        ax3.set_xlim(0, self.history_length)
        ax3.set_ylim(-15, 15)
        ax3.grid(True, alpha=0.2)
        ax3.tick_params(colors='white')
        if self.gt_pitch_history and len(self.pitch_history) > 1:
            ax3.legend(loc='upper right', fontsize=8)
        
        # Convert plot to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        # ARGB format has 4 channels, we need to handle this correctly
        w, h = canvas.get_width_height()
        dashboard_img = buf.reshape((h, w, 4))
        # Convert ARGB to RGB by dropping the alpha channel
        dashboard_img = dashboard_img[:, :, 1:]
        plt.close(fig)
        
        return cv2.cvtColor(dashboard_img, cv2.COLOR_RGB2BGR)
    
    def find_focus_of_expansion(self, flow, gray_frame=None):
        """Find FOE using weighted least squares method."""
        h, w = flow.shape[:2]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Flatten arrays
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        flow_x = flow[:, :, 0].flatten()
        flow_y = flow[:, :, 1].flatten()
        
        # Calculate flow magnitude
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        
        # Create weight based on flow magnitude (stronger flows are more reliable)
        # Use sqrt to prevent very large flows from dominating
        weights = np.sqrt(magnitude)
        
        # Filter out very small flows
        min_magnitude = 0.5
        mask = magnitude > min_magnitude
        
        # Exclude top portion (sky) which often has noise
        sky_mask = y_coords.flatten() > h * self.sky_exclusion_ratio
        mask = mask & sky_mask
        
        # If we have the grayscale frame, use it to filter dark regions
        if gray_frame is not None:
            brightness_flat = gray_frame.flatten()
            # Exclude very dark regions (brightness < 40) where flow is unreliable
            brightness_mask = brightness_flat > 40
            mask = mask & brightness_mask
            
            # Also reduce weights in darker regions
            # Normalize brightness to 0-1 range and use as additional weight
            brightness_weight = np.clip(brightness_flat / 255.0, 0.1, 1.0)
            weights = weights * brightness_weight
        
        if np.sum(mask) < 10:
            return w/2, h/2
        
        # Apply mask and get weights
        x_flat = x_flat[mask]
        y_flat = y_flat[mask]
        flow_x = flow_x[mask]
        flow_y = flow_y[mask]
        weights = weights[mask]
        
        # Weighted least squares solution
        # Weight both sides of the equation
        A = np.column_stack([flow_y, -flow_x])
        b = x_flat * flow_y - y_flat * flow_x
        
        # Apply weights efficiently without creating diagonal matrix
        # Multiply each row by its weight
        A_weighted = A * weights[:, np.newaxis]
        b_weighted = b * weights
        
        try:
            # Solve weighted least squares
            foe, _, _, _ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
            foe_x, foe_y = foe
            
            # Ensure FOE is within image bounds
            foe_x = np.clip(foe_x, 0, w-1)
            foe_y = np.clip(foe_y, 0, h-1)
        except:
            foe_x, foe_y = w/2, h/2
        
        return foe_x, foe_y
    
    def smooth_foe(self, foe_x, foe_y, brightness_change_detected=False):
        """Apply temporal smoothing to FOE position to reduce jitter and spikes."""
        # Adjust smoothing based on lighting conditions
        if brightness_change_detected:
            # Use heavier smoothing during lighting changes
            effective_alpha = self.smoothing_alpha * 0.5
            effective_threshold = self.outlier_threshold * 0.7
        else:
            effective_alpha = self.smoothing_alpha
            effective_threshold = self.outlier_threshold
        
        # If we have history, check for outliers
        if len(self.foe_history_x) > 0:
            # Calculate distance from median of recent positions
            median_x = np.median(list(self.foe_history_x))
            median_y = np.median(list(self.foe_history_y))
            distance = np.sqrt((foe_x - median_x)**2 + (foe_y - median_y)**2)
            
            # If it's an outlier, use the median instead
            if distance > effective_threshold:
                foe_x = median_x
                foe_y = median_y
        
        # Add to history
        self.foe_history_x.append(foe_x)
        self.foe_history_y.append(foe_y)
        
        # Apply temporal smoothing
        if len(self.foe_history_x) > 1:
            # Exponential moving average
            prev_x = list(self.foe_history_x)[-2]
            prev_y = list(self.foe_history_y)[-2]
            
            smoothed_x = effective_alpha * foe_x + (1 - effective_alpha) * prev_x
            smoothed_y = effective_alpha * foe_y + (1 - effective_alpha) * prev_y
            
            return smoothed_x, smoothed_y
        else:
            return foe_x, foe_y
    
    def process_frame(self, prev_frame, curr_frame):
        """Process a single frame pair and create visualization."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect brightness changes
        curr_brightness = np.mean(gray2)
        brightness_change_detected = False
        if self.prev_brightness is not None:
            brightness_diff = abs(curr_brightness - self.prev_brightness)
            brightness_change_detected = brightness_diff > self.brightness_change_threshold
        self.prev_brightness = curr_brightness
        
        # Skip CLAHE - it amplifies noise in dark regions
        # Instead, just apply mild Gaussian blur to reduce noise
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 1.0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 1.0)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **self.flow_params)
        
        # Find FOE using weighted least squares (pass gray2 for brightness-based filtering)
        raw_foe_x, raw_foe_y = self.find_focus_of_expansion(flow, gray2)
        
        # Apply temporal smoothing to reduce jitter (with adaptive smoothing during brightness changes)
        foe_x, foe_y = self.smooth_foe(raw_foe_x, raw_foe_y, brightness_change_detected)
        
        # Convert to angles
        h, w = gray1.shape
        cx, cy = w/2, h/2
        raw_yaw = np.arctan((foe_x - cx) / self.focal_length)
        raw_pitch = np.arctan(-(foe_y - cy) / self.focal_length)
        
        # Apply additional smoothing to angles
        self.pitch_smooth.append(raw_pitch)
        self.yaw_smooth.append(raw_yaw)
        
        # Use median filter on angles to remove spikes
        if len(self.pitch_smooth) >= 3:
            pitch = np.median(list(self.pitch_smooth))
            yaw = np.median(list(self.yaw_smooth))
        else:
            pitch = raw_pitch
            yaw = raw_yaw
        
        # Estimate relative speed from average flow magnitude
        speed_estimate = np.mean(np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2))
        
        # Update history
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)
        self.speed_history.append(speed_estimate)
        
        # Update ground truth history if available
        if self.ground_truth is not None and self.frame_idx < len(self.ground_truth):
            self.gt_pitch_history.append(self.ground_truth[self.frame_idx, 0])
            self.gt_yaw_history.append(self.ground_truth[self.frame_idx, 1])
        
        # Increment frame index
        self.frame_idx += 1
        
        # Create visualizations
        flow_hsv = self.create_flow_hsv_visualization(flow)
        flow_arrows = self.create_flow_arrow_visualization(curr_frame, flow, foe_x, foe_y, raw_foe_x, raw_foe_y)
        motion_trail = self.create_motion_trail(curr_frame)
        dashboard = self.create_dashboard(pitch, yaw, speed_estimate)
        
        return {
            'flow_hsv': flow_hsv,
            'flow_arrows': flow_arrows,
            'motion_trail': motion_trail,
            'dashboard': dashboard,
            'pitch': pitch,
            'yaw': yaw,
            'foe': (foe_x, foe_y),
            'raw_foe': (raw_foe_x, raw_foe_y),
            'brightness_change': brightness_change_detected
        }

def create_combined_visualization(visualizer, prev_frame, curr_frame, layout='grid'):
    """Create combined visualization with multiple views."""
    results = visualizer.process_frame(prev_frame, curr_frame)
    
    h, w = curr_frame.shape[:2]
    
    if layout == 'grid':
        # 2x2 grid layout
        top_row = np.hstack([curr_frame, results['flow_arrows']])
        bottom_row = np.hstack([results['flow_hsv'], results['motion_trail']])
        grid = np.vstack([top_row, bottom_row])
        
        # Resize dashboard to fit
        dashboard_h = grid.shape[0]
        dashboard_w = int(results['dashboard'].shape[1] * dashboard_h / results['dashboard'].shape[0])
        dashboard_resized = cv2.resize(results['dashboard'], (dashboard_w, dashboard_h))
        
        # Combine grid with dashboard
        combined = np.hstack([grid, dashboard_resized])
        
    elif layout == 'horizontal':
        # Horizontal strip layout
        combined = np.hstack([curr_frame, results['flow_arrows'], results['flow_hsv']])
        
    elif layout == 'vertical':
        # Vertical layout with dashboard
        main_vis = np.vstack([curr_frame, results['flow_arrows']])
        
        # Resize dashboard to match width
        dashboard_w = main_vis.shape[1] // 3
        dashboard_h = int(results['dashboard'].shape[0] * dashboard_w / results['dashboard'].shape[1])
        dashboard_resized = cv2.resize(results['dashboard'], (dashboard_w, dashboard_h))
        
        # Pad dashboard to match height
        if dashboard_resized.shape[0] < main_vis.shape[0]:
            pad = main_vis.shape[0] - dashboard_resized.shape[0]
            dashboard_resized = np.vstack([
                dashboard_resized,
                np.zeros((pad, dashboard_w, 3), dtype=np.uint8)
            ])
        
        combined = np.hstack([main_vis, dashboard_resized])
    
    # Add text overlay
    cv2.putText(combined, f"Yaw: {np.degrees(results['yaw']):+.1f}° | Pitch: {np.degrees(results['pitch']):+.1f}°", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show if brightness change detected
    if 'brightness_change' in results and results['brightness_change']:
        cv2.putText(combined, "BRIGHTNESS CHANGE DETECTED", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    
    return combined, results

def process_video_with_visualization(video_path, output_video_path=None, 
                                   output_data_path=None, layout='grid',
                                   show_live=True, save_video=True, ground_truth_path=None):
    """
    Process video with comprehensive visualization.
    
    Args:
        video_path: Input video path
        output_video_path: Output visualization video path
        output_data_path: Output data file path
        layout: Visualization layout ('grid', 'horizontal', 'vertical')
        show_live: Show live preview
        save_video: Save visualization video
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Load ground truth if provided
    ground_truth = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        ground_truth = np.loadtxt(ground_truth_path)
        print(f"Loaded ground truth data from {ground_truth_path}")
    
    visualizer = FlowVisualizer(focal_length=910, ground_truth=ground_truth)
    predictions = []
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
    # Setup video writer if needed
    if save_video and output_video_path:
        # Get dimensions for the combined visualization
        ret, temp_frame = cap.read()
        if ret:
            temp_combined, _ = create_combined_visualization(visualizer, prev_frame, temp_frame, layout)
            h, w = temp_combined.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)  # Reset to frame 1
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    frame_count = 0
    
    print("Processing video... Press 'q' to quit, 'p' to pause")
    paused = False
    
    while True:
        if not paused:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            try:
                # Create visualization
                combined, results = create_combined_visualization(visualizer, prev_frame, curr_frame, layout)
                
                # Save predictions
                predictions.append([results['pitch'], results['yaw']])
                
                # Write to output video
                if save_video and output_video_path:
                    out.write(combined)
                
                prev_frame = curr_frame
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"  Processed {frame_count} frames...")
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                break
        
        # Show live preview
        if show_live and 'combined' in locals():
            display = cv2.resize(combined, (combined.shape[1]//2, combined.shape[0]//2))
            cv2.imshow('Flow Analysis Visualization', display)
            
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
    
    # Cleanup
    print("\nCleaning up...")
    try:
        cap.release()
        if save_video and output_video_path and 'out' in locals():
            out.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Warning during cleanup: {e}")
    
    # Save predictions
    if output_data_path and predictions:
        try:
            predictions_array = np.array(predictions)
            # Save in the same format as ground truth (space-separated, no header)
            np.savetxt(output_data_path, predictions_array, delimiter=' ', fmt='%.18e')
            print(f"Saved predictions to {output_data_path}")
        except Exception as e:
            print(f"Error saving predictions: {e}")
    
    print(f"Processing complete! Processed {frame_count} frames.")
    return np.array(predictions) if predictions else np.array([])

def process_all_videos():
    """Process all videos in labeled and unlabeled folders."""
    import subprocess
    
    # Create output directory for predictions
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process labeled videos (0-4)
    print("Processing labeled videos...")
    for i in range(5):
        video_path = f"labeled/{i}.hevc"
        ground_truth_path = f"labeled/{i}.txt"
        output_path = f"{output_dir}/{i}.txt"
        
        print(f"\nProcessing {video_path}...")
        predictions = process_video_with_visualization(
            video_path,
            output_video_path=None,  # Don't save visualization videos
            output_data_path=output_path,
            layout='grid',
            show_live=False,  # Run headless for batch processing
            save_video=False,
            ground_truth_path=ground_truth_path
        )
        
        if predictions is not None and len(predictions) > 0:
            print(f"  Saved {len(predictions)} frames to {output_path}")
            print(f"  Average pitch: {np.degrees(np.mean(predictions[:, 0])):.2f}°")
            print(f"  Average yaw: {np.degrees(np.mean(predictions[:, 1])):.2f}°")
    
    # Process unlabeled videos (5-9)
    print("\n\nProcessing unlabeled videos...")
    for i in range(5, 10):
        video_path = f"unlabeled/{i}.hevc"
        output_path = f"{output_dir}/{i}.txt"
        
        print(f"\nProcessing {video_path}...")
        predictions = process_video_with_visualization(
            video_path,
            output_video_path=None,
            output_data_path=output_path,
            layout='grid',
            show_live=False,
            save_video=False,
            ground_truth_path=None
        )
        
        if predictions is not None and len(predictions) > 0:
            print(f"  Saved {len(predictions)} frames to {output_path}")
            print(f"  Average pitch: {np.degrees(np.mean(predictions[:, 0])):.2f}°")
            print(f"  Average yaw: {np.degrees(np.mean(predictions[:, 1])):.2f}°")
    
    # Run evaluation on labeled data
    print("\n\nRunning evaluation on labeled videos...")
    try:
        result = subprocess.run(
            ["python", "eval.py", output_dir],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        print(f"Error output: {e.stderr}")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        # Batch process all videos
        process_all_videos()
    else:
        # Single video processing (for testing/visualization)
        video_path = "labeled/0.hevc"
        output_video = "flow_analysis_visualization.mp4"
        output_data = "predictions.txt"
        
        # Check if this is a labeled video and load ground truth
        ground_truth_path = None
        if video_path.startswith("labeled/"):
            # Extract the number from the video path
            video_num = os.path.splitext(os.path.basename(video_path))[0]
            ground_truth_path = f"labeled/{video_num}.txt"
        
        predictions = process_video_with_visualization(
            video_path,
            output_video_path=output_video,
            output_data_path=output_data,
            layout='grid',
            show_live=True,
            save_video=True,
            ground_truth_path=ground_truth_path
        )
        
        print(f"\nResults summary:")
        print(f"Total frames: {len(predictions)}")
        if len(predictions) > 0:
            print(f"Average yaw: {np.degrees(np.mean(predictions[:, 1])):.2f}°")
            print(f"Average pitch: {np.degrees(np.mean(predictions[:, 0])):.2f}°")
            print(f"Yaw std dev: {np.degrees(np.std(predictions[:, 1])):.2f}°")
            print(f"Pitch std dev: {np.degrees(np.std(predictions[:, 0])):.2f}°")
            
            # If we have ground truth, compute error metrics
            if ground_truth_path and os.path.exists(ground_truth_path):
                gt = np.loadtxt(ground_truth_path)[:len(predictions)]
                mse_pitch = np.mean((predictions[:, 0] - gt[:, 0])**2)
                mse_yaw = np.mean((predictions[:, 1] - gt[:, 1])**2)
                print(f"\nError metrics vs ground truth:")
                print(f"MSE Pitch: {mse_pitch:.6f} rad² ({np.degrees(np.sqrt(mse_pitch)):.2f}° RMSE)")
                print(f"MSE Yaw: {mse_yaw:.6f} rad² ({np.degrees(np.sqrt(mse_yaw)):.2f}° RMSE)")
