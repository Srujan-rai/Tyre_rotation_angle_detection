import cv2
import numpy as np

def calculate_rotation_angle(initial_image, current_frame):
    # Convert images to grayscale
    initial_gray = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Use template matching
    result = cv2.matchTemplate(current_gray, initial_gray, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Calculate the center coordinates of the template in the current frame
    template_center = (max_loc[0] + initial_image.shape[1] // 2, max_loc[1] + initial_image.shape[0] // 2)

    # Calculate the displacement vector
    displacement_vector = (template_center[0] - current_frame.shape[1] // 2, template_center[1] - current_frame.shape[0] // 2)

    # Calculate the angle of rotation using arctangent
    angle_of_rotation_radians = np.arctan2(displacement_vector[1], displacement_vector[0])

    # Convert radians to degrees
    angle_of_rotation_degrees = np.degrees(angle_of_rotation_radians)

    return angle_of_rotation_degrees

# Capture video from camera (adjust the camera index if needed)
cap = cv2.VideoCapture(0)

# Read the initial image
initial_image_path = 'images.jpeg'
initial_image = cv2.imread(initial_image_path)

while True:
    # Capture current frame
    ret, frame = cap.read()

    # Break the loop if video capture fails
    if not ret:
        print("Error capturing video feed")
        break

    # Calculate the rotation angle
    rotation_angle = calculate_rotation_angle(initial_image, frame)

    # Display the frame with rotation angle
    cv2.putText(frame, f"Rotation Angle: {rotation_angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video Feed', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
