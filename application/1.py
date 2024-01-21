import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class TireRotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tire Rotation Angle Detection")

        self.capture_button = tk.Button(root, text="Capture Image", command=self.capture_image)
        self.capture_button.pack(pady=10)

        self.angle_label = tk.Label(root, text="Rotation Angle: ")
        self.angle_label.pack(pady=10)

        self.initial_image = None

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        _, self.frame = self.cap.read()

        # Convert frame to PhotoImage format for display in Tkinter
        self.image_tk = self.convert_frame_to_photoimage(self.frame)

        # Display the current frame
        self.image_label = tk.Label(self.root, image=self.image_tk)
        self.image_label.image = self.image_tk
        self.image_label.pack(pady=10)

        # Start continuous update
        self.root.after(10, self.update)

    def capture_image(self):
        # Capture initial image from camera
        _, self.initial_image = self.cap.read()

    def update(self):
        # Capture current frame from camera
        _, current_frame = self.cap.read()

        # If an initial image is captured, calculate the rotation angle
        if self.initial_image is not None:
            rotation_angle = self.calculate_rotation_angle_template_matching(self.initial_image, current_frame)

            # Update the angle label
            self.angle_label.config(text=f"Rotation Angle: {rotation_angle:.2f} degrees")

        # Convert frame to PhotoImage format for display in Tkinter
        self.image_tk = self.convert_frame_to_photoimage(current_frame)

        # Update the displayed image
        self.image_label.configure(image=self.image_tk)
        self.image_label.image = self.image_tk

        # Continue updating the video feed and angle continuously
        self.root.after(10, self.update)

    def calculate_rotation_angle_template_matching(self, initial_image, current_frame):
        # Convert images to grayscale
        initial_gray = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Use template matching
        result = cv2.matchTemplate(current_gray, initial_gray, cv2.TM_CCOEFF_NORMED)

        # Find the location of the best match
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Calculate the displacement vector
        displacement_vector = (
            max_loc[0] + initial_image.shape[1] // 2 - current_frame.shape[1] // 2,
            max_loc[1] + initial_image.shape[0] // 2 - current_frame.shape[0] // 2
        )

        # Calculate the angle of rotation using arctangent
        angle_of_rotation_radians = np.arctan2(displacement_vector[1], displacement_vector[0])

        # Convert radians to degrees
        angle_of_rotation_degrees = np.degrees(angle_of_rotation_radians)

        return angle_of_rotation_degrees

    def convert_frame_to_photoimage(self, frame):
        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to fit the window
        height, width, _ = frame.shape
        new_width = 600
        new_height = int((new_width / width) * height)
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

        # Convert frame to PhotoImage format
        image_tk = ImageTk.PhotoImage(Image.fromarray(frame_resized))

        return image_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = TireRotationApp(root)
    root.mainloop()
