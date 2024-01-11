import cv2
import numpy as np

# Function to track tire rotation using Shi-Tomasi corner detection and Lucas-Kanade optical flow
def track_tire_rotation(camera_index=0):
    cap = cv2.VideoCapture(camera_index)  # Use camera index as input for live video feed

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read the first frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Lucas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Check if p1 is not None before proceeding
        if p1 is None:
            continue

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

        img = cv2.add(frame, mask)

        cv2.imshow('Tire Rotation Tracking', img)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(30) & 0xFF == 27:
            break

        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()

# Use camera index 0 by default (change it if your camera has a different index)
track_tire_rotation()
