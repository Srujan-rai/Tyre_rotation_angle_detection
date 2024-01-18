import cv2
import numpy as np

def calculate_rotation_angle(initial_image_path, current_image_path):
    # Read the images
    initial_image = cv2.imread(initial_image_path)
    current_image = cv2.imread(current_image_path)

    # Convert images to grayscale
    initial_gray = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # Use a small region of interest (ROI) as the template (adjust as needed)
    template_size = (50, 50)
    roi_initial = initial_gray[100:100 + template_size[1], 100:100 + template_size[0]]
    cv2.imshow("roi",roi_initial)
    cv2.waitKey(0)
    # Perform template matching
    result = cv2.matchTemplate(current_gray, roi_initial, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Calculate the center coordinates of the template in the current image
    template_center = (max_loc[0] + template_size[0] // 2, max_loc[1] + template_size[1] // 2)

    # Calculate the displacement vector
    displacement_vector = (template_center[0] - initial_image.shape[1] // 2, template_center[1] - initial_image.shape[0] // 2)

    # Calculate the angle of rotation using arctangent
    angle_of_rotation_radians = np.arctan2(displacement_vector[1], displacement_vector[0])

    # Convert radians to degrees
    angle_of_rotation_degrees = np.degrees(angle_of_rotation_radians)

    return angle_of_rotation_degrees

# Example usage:
initial_image_path = 'images.jpeg'
current_image_path = 'images1.jpeg'

rotation_angle = calculate_rotation_angle(initial_image_path, current_image_path)

print(f"Angle of rotation: {rotation_angle} degrees")
