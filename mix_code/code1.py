from ultralytics import YOLO
import cv2
import numpy as np
import math

# Initialize YOLO model and camera
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

# Parameters for tire rotation tracking
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create windows for displaying frames
cv2.namedWindow('Original with Bounding Box', cv2.WINDOW_NORMAL)

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Initialize points for tire rotation tracking
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6)

    for r in results:
        boxes = r.boxes  # Accessing the bounding box coordinates
        
        if boxes is not None and len(boxes.xyxy) > 0:
            for pred in boxes.xyxy:
                if pred.ndimension() > 0:
                    x_min, y_min, x_max, y_max = pred.tolist()  # Extract the coordinates as a list

                    # Draw bounding box on the original frame
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                    # Crop the region within the bounding box
                    cropped_object = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

                    # Tire rotation tracking within the bounding box
                    frame_gray = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2GRAY)

                    # Ensure that the size of p0 is consistent
                    if p0 is not None and p0.shape[0] > 0:
                        # Calculate optical flow for tire rotation tracking
                        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                        # Check if points are not None before proceeding
                        if p1 is not None:
                            # Select good points
                            good_new = p1[st == 1]
                            good_old = p0[st == 1]

                            # Draw the tracks
                            for i, (new, old) in enumerate(zip(good_new, good_old)):
                                a, b = new.ravel()
                                c, d = old.ravel()
                                mask = cv2.line(mask, (int(a) + int(x_min), int(b) + int(y_min)),
                                                (int(c) + int(x_min), int(d) + int(y_min)), (0, 255, 0), 2)
                                cropped_object = cv2.circle(cropped_object, (int(a), int(b)), 5, (0, 255, 0), -1)

                            img = cv2.add(cropped_object, mask)

                            cv2.imshow('Tire Rotation Tracking', img)

                        # Update previous frame and points for tire rotation tracking
                        old_gray = frame_gray.copy()
                        p0 = good_new.reshape(-1, 1, 2)

    # Display the original frame with bounding box
    cv2.imshow('Original with Bounding Box', frame)

    # Wait for a key press and close windows if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
cap.release()
