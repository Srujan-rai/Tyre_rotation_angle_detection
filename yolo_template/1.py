import cv2
import numpy as np
from ultralytics import YOLO

def calculate_rotation_angle(initial_image, current_frame):
    # Convert images to grayscale
    initial_gray = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Ensure that the template image is smaller or equal in size to the current frame
    initial_gray = cv2.resize(initial_gray, (current_gray.shape[1], current_gray.shape[0]))

    # Use template matching
    result = cv2.matchTemplate(current_gray, initial_gray, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Calculate the center coordinates of the template in the current frame
    template_center = (max_loc[0] + initial_gray.shape[1] // 2, max_loc[1] + initial_gray.shape[0] // 2)

    # Calculate the displacement vector
    displacement_vector = (template_center[0] - current_frame.shape[1] // 2, template_center[1] - current_frame.shape[0] // 2)

    # Calculate the angle of rotation using arctangent
    angle_of_rotation_radians = np.arctan2(displacement_vector[1], displacement_vector[0])

    # Convert radians to degrees
    angle_of_rotation_degrees = np.degrees(angle_of_rotation_radians)

    return angle_of_rotation_degrees

# Initialize YOLO model and camera
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

# Read the initial image
initial_image_path = 'images.jpeg'
initial_image = cv2.imread(initial_image_path)

# Create windows for displaying frames
cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original with Bounding Box', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection using YOLO
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

                    # Perform template matching inside the detected object
                    if initial_image is not None:
                        rotation_angle = calculate_rotation_angle(initial_image, cropped_object)

                        # Display the rotation angle inside the bounding box
                        cv2.putText(frame, f"Rotation Angle: {rotation_angle:.2f} degrees",
                                    (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        print(rotation_angle)
                    # Display the cropped object
                    cv2.imshow('Detected Object', cropped_object)

        # Display the original frame with bounding box
        cv2.imshow('Original with Bounding Box', frame)

        # Wait for a key press and close windows if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
