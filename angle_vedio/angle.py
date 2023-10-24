# Import the required modules
import cv2
import math
import numpy as np

# Define a function to calculate the slope of a line
def slope(p1, p2):
    # p1 and p2 are tuples of (x, y) coordinates
    # Return the slope of the line passing through p1 and p2
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

# Read the image and convert it to grayscale
img = cv2.imread("input_img.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to find the contours of the object
edges = cv2.Canny(gray, 100, 200)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Find the minimum area bounding rectangle for the largest contour
# This assumes that the object is the largest contour in the image
# You may need to sort the contours by area and choose the one you want
cnt = contours[0]
rect = cv2.minAreaRect(cnt)

# Get the four vertices of the rectangle
box = cv2.boxPoints(rect)
box = np.int0(box)

# Draw the rectangle on the original image
cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

# Get the angle of the rectangle
angle = rect[-1]

# Correct the angle if it is negative
if angle < 0:
    angle = 90 + angle

# Print the angle on the original image
cv2.putText(img, "Angle: {:.2f}".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Show the original image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
