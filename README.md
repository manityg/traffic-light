Traffic Light Density-Based Timing Adjustment
This project uses video analysis to measure vehicular density at a traffic intersection and dynamically adjusts green light timings to optimize traffic flow. By leveraging computer vision, this program calculates the density of vehicles in real-time and adjusts green light duration for each traffic light accordingly.

Features
Vehicular Density Detection: Analyzes video feeds of traffic lights to detect vehicular density in specific regions.
Dynamic Green Light Timing: Adjusts green light timing based on real-time density measurements.
Flexible Parameters: Supports adjustments for frame sampling interval, total observation duration, thresholding, and kernel size.

Requirements
Python 3.x
OpenCV (cv2)
NumPy (numpy)
Install dependencies via:

pip install opencv-python-headless numpy
Project Structure
video_paths: Paths to traffic light videos, replace with actual paths to analyze specific video feeds.
Density Detection Functions: Functions to extract regions of interest (ROI), detect vehicular density, and calculate average densities over time.
Usage
Set Up Video Paths: Update the video_paths list with paths to your traffic light videos.

Configure Parameters: Adjust parameters like interval_duration, total_duration, threshold, and kernel_size if needed.

Run the Script: Execute the script to calculate vehicular density and adjust green light timings.

python traffic_density_adjustment.py
View Results: The program displays the calculated density for each frame and computes optimized green light times for each traffic direction.

Example Output
The script outputs adjusted green light durations based on density calculations from each video feed:

Adjusted green light time for light 1: 32.50 seconds
Adjusted green light time for light 2: 28.00 seconds
...
Parameters
interval_duration: Frame sampling interval in seconds.
total_duration: Total observation duration for each video.
roi_coordinates: Defines the region of interest for vehicle detection.
threshold: Threshold value for binary conversion in density detection.
kernel_size: Kernel size for morphological operations to refine detection accuracy.

License

This project is licensed under the MIT License.
