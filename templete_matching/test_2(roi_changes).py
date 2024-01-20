import cv2
import numpy as np

def calculate_rotation_angle(initial_image_path, current_image_path):
    # Read the images
    initial_image = cv2.imread(initial_image_path)
    current_image = cv2.imread(current_image_path)

    # Convert images to grayscale
    initial_gray = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    # Use SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT for the initial image
    keypoints1, descriptors1 = sift.detectAndCompute(initial_gray, None)

    # Find the keypoints and descriptors with SIFT for the current image
    keypoints2, descriptors2 = sift.detectAndCompute(current_gray, None)

    # Use the FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches using K-Nearest Neighbors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate the rotation angle using the matched keypoints
    angle_sum = 0.0
    for match in good_matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        angle_sum += np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

    # Average the angles
    average_angle = np.degrees(angle_sum / len(good_matches))

    return average_angle

# Example usage:
initial_image_path = 'images.jpeg'
current_image_path = 'images1.jpeg'

rotation_angle = calculate_rotation_angle(initial_image_path, current_image_path)

print(f"Angle of rotation: {rotation_angle} degrees")
