import cv2
import numpy as np

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use the Hough Line Transform to detect lines in the frame
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is not None:
        for rho, theta in lines[0]:
            # Convert polar coordinates to Cartesian coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Calculate the angle of rotation (in degrees)
            angle = np.degrees(theta)

            # Draw the line on the original frame
            cv2.line(frame, (int(x0 - 1000 * b), int(y0 + 1000 * a)),
                     (int(x0 + 1000 * b), int(y0 - 1000 * a)), (0, 0, 255), 2)

            print(f"Angle of rotation: {angle} degrees")

    # Display the frame with the detected line
    cv2.imshow("Live Video", frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
