


import cv2
import numpy as np
import time

# Paths to video files for each traffic light
video_paths = ["test1.mp4", "test2.mp4", "test3.mp4", "test4.mp4"]  # Replace with your actual video paths
interval_duration = 15  # Interval for sampling frames in seconds
total_duration = 120  # Total observation duration in seconds
num_samples = total_duration // interval_duration
roi_coordinates = (0, 0, 945, 768)  # Update as needed based on your video frame dimensions

# Default parameters
threshold = 20
kernel_size = 5

def extract_road_roi(frame, roi_coordinates):
    x, y, w, h = roi_coordinates
    return frame[y:y+h, x:x+w]

def detect_vehicular_density_adaptive(roi, blur_level, threshold, min_blob_area, kernel_size):
    # Adaptive background subtraction
    blurred_background = cv2.GaussianBlur(roi, (blur_level, blur_level), 0)
    background_subtracted = cv2.absdiff(roi, blurred_background)
    _, binary = cv2.threshold(background_subtracted, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Contour detection
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_roi_area = roi.shape[0] * roi.shape[1]
    
    # Calculate blob area
    blob_area_sum = sum(cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > min_blob_area)
    
    # Density calculation
    density = (blob_area_sum / total_roi_area) * 100
    return density

def calculate_average_density_from_video(video_path, roi_coordinates, num_samples, interval_duration):
    densities = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {video_path}. Please check the file path.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps * interval_duration)  # Calculate number of frames to skip for each interval

    print(f"Processing video: {video_path}")
    
    for sample in range(num_samples):
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached for {video_path} at sample {sample+1}.")
            break

        # Convert frame to grayscale and extract ROI
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = extract_road_roi(gray_frame, roi_coordinates)

        # Dynamic parameter adjustment based on lighting conditions
        if np.mean(roi) < 100:  # Darker frames, possibly nighttime
            blur = 51
            blob_area = 400
        else:  # Brighter frames, possibly daytime
            blur = 31
            blob_area = 200

        # Calculate density for this frame
        density = detect_vehicular_density_adaptive(
            roi,
            blur_level=blur,
            threshold=threshold,
            min_blob_area=blob_area,
            kernel_size=kernel_size
        )
        
        densities.append(density)
        
        # Print the occupancy (density) of the current frame
        print(f"Frame {sample+1} of {video_path}: Occupancy = {density:.2f}%")
        
        # Optional: Display the ROI with density annotation
        annotated_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        cv2.putText(annotated_roi, f'Density: {density:.2f}%', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f'ROI - {video_path}', annotated_roi)
        
        # Wait for a short period to display the frame (adjust as needed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Skip to the next interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip_frames)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Return the average density for the video
    average_density = sum(densities) / len(densities) if densities else 0
    print(f"Average density for {video_path}: {average_density:.2f}%\n")
    return average_density

# Calculate average densities for each traffic light video
average_densities = [calculate_average_density_from_video(path, roi_coordinates, num_samples, interval_duration) for path in video_paths]

# Algorithm for green light timing using average densities
Omax = 100
Tmin = 25  # Minimum green light time in seconds
Total = 120  # Total green light cycle duration in seconds

# Normalize each density
normalized_densities = [density / Omax for density in average_densities]

# Calculate total normalized occupancy
Ototal = sum(normalized_densities)

# Calculate proportion of time for each direction
proportions = [On / Ototal for On in normalized_densities]

# Calculate adjusted green light times for each direction
adjusted_times = [Tmin + (Total - (Tmin * len(video_paths))) * Pi for Pi in proportions]

# Print results
for i, time_sec in enumerate(adjusted_times, start=1):
    print(f"Adjusted green light time for light {i}: {time_sec:.2f} seconds")
