import cv2
import numpy as np

videos = ["test1.mp4", "test2.mp4", "test3.mp4", "test4.mp4"]
interval = 15
duration = 120
samples = duration // interval
roi = (0, 0, 945, 768)

thresh = 20
kernel_size = 5

def get_roi(frame, roi):
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]

def calc_density(roi, blur, thresh, min_area, kernel_size):
    blurred = cv2.GaussianBlur(roi, (blur, blur), 0)
    subtracted = cv2.absdiff(roi, blurred)
    _, binary = cv2.threshold(subtracted, thresh, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = roi.shape[0] * roi.shape[1]
    
    area_sum = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > min_area)
    
    density = (area_sum / total_area) * 100
    return density

def avg_density(video, roi, samples, interval):
    densities = []
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {video}.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps * interval)

    for sample in range(samples):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_frame = get_roi(gray, roi)

        if np.mean(roi_frame) < 100:
            blur = 51
            min_area = 400
        else:
            blur = 31
            min_area = 200

        density = calc_density(roi_frame, blur, thresh, min_area, kernel_size)
        
        densities.append(density)
        
        print(f"Frame {sample+1} of {video}: Occupancy = {density:.2f}%")
        
        annotated = cv2.cvtColor(roi_frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(annotated, f'Density: {density:.2f}%', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f'ROI - {video}', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip_frames)
    
    cap.release()
    cv2.destroyAllWindows()
    
    avg_density = sum(densities) / len(densities) if densities else 0
    print(f"Average density for {video}: {avg_density:.2f}%\n")
    return avg_density

avg_densities = [avg_density(path, roi, samples, interval) for path in videos]

Omax = 100
Tmin = 25
Total = 120

norm_densities = [d / Omax for d in avg_densities]

Ototal = sum(norm_densities)

proportions = [d / Ototal for d in norm_densities]

adjusted_times = [Tmin + (Total - (Tmin * len(videos))) * p for p in proportions]

for i, time_sec in enumerate(adjusted_times, start=1):
    print(f"Adjusted green light time for light {i}: {time_sec:.2f} seconds")