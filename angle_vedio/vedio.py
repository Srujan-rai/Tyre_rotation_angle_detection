import cv2
import math

# Function to calculate the angle between two lines
def calculate_angle(line1, line2):
    angle1 = math.atan2(line1[1] - line1[3], line1[0] - line1[2])
    angle2 = math.atan2(line2[1] - line2[3], line2[0] - line2[2])
    angle = math.degrees(abs(angle1 - angle2))
    return angle

# Capture video from your webcam (you can also specify a video file path)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Perform Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if len(lines) >= 2:
            angle = calculate_angle(lines[0][0], lines[1][0])
            cv2.putText(frame, f'Angle: {angle:.2f} degrees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
