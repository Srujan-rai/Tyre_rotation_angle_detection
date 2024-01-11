import cv2
import numpy as np
import math

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
    
    # Initialize points for the first tracked feature
    p0_1 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Initialize points for the second tracked feature with an offset from p0_1
    offset = np.array([[[50, 50]]], dtype=np.float32)
    p0_2 = p0_1 + offset

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow for the first point
        p1_1, st_1, err_1 = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_1, None, **lk_params)

        # Calculate optical flow for the second point
        p1_2, st_2, err_2 = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_2, None, **lk_params)

        # Check if either set of points is not None before proceeding
        if p1_1 is None or p1_2 is None:
            continue

        # Select good points for the first point
        good_new_1 = p1_1[st_1 == 1]
        good_old_1 = p0_1[st_1 == 1]

        # Select good points for the second point
        good_new_2 = p1_2[st_2 == 1]
        good_old_2 = p0_2[st_2 == 1]

        # Draw the tracks for the first point
        for i, (new, old) in enumerate(zip(good_new_1, good_old_1)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

        # Draw the tracks for the second point
        for i, (new, old) in enumerate(zip(good_new_2, good_old_2)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Calculate the angle between the two vectors formed by the points
        angle_rad = math.atan2(good_new_2[0, 1] - good_new_1[0, 1], good_new_2[0, 0] - good_new_1[0, 0])
        angle_deg = math.degrees(angle_rad)
        print("Angle: {:.2f} degrees".format(angle_deg))

        img = cv2.add(frame, mask)

        cv2.imshow('Tire Rotation Tracking', img)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(30) & 0xFF == 27:
            break

        # Update previous frame and points for the first point
        old_gray = frame_gray.copy()
        p0_1 = good_new_1.reshape(-1, 1, 2)

        # Update points for the second point
        p0_2 = good_new_2.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()

# Use camera index 0 by default (change it if your camera has a different index)
track_tire_rotation()
