from ultralytics import YOLO
import cv2

# Initialize YOLO model and camera
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

# Create windows for displaying frames
cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original with Bounding Box', cv2.WINDOW_NORMAL)

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

                    # Display the cropped object
                    cv2.imshow('Detected Object', cropped_object)

        # Display the original frame with bounding box
        cv2.imshow('Original with Bounding Box', frame)
        
        # Wait for a key press and close windows if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
