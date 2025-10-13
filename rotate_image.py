"""
Interactive Image Rotation Script
Loads an image, allows interactive rotation, and provides save option.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import os

class ImageRotator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from '{image_path}'")

        self.angle = 0
        self.rotated_image = self.original_image.copy()


        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle('Image Rotation Tool')


        self.ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        self.ax1.set_title('Original Image')
        self.ax1.axis('off')


        self.rotated_rgb = cv2.cvtColor(self.rotated_image, cv2.COLOR_BGR2RGB)
        self.im2 = self.ax2.imshow(self.rotated_rgb)
        self.ax2.set_title(f'Rotated Image (0°)')
        self.ax2.axis('off')


        self.slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
        self.slider = widgets.Slider(self.slider_ax, 'Rotation (°)', -180, 180, valinit=0, valstep=1)
        self.slider.on_changed(self.update_rotation)


        self.save_ax = plt.axes([0.8, 0.9, 0.1, 0.05])
        self.save_button = widgets.Button(self.save_ax, 'Save')
        self.save_button.on_clicked(self.save_image)

        plt.tight_layout()

    def update_rotation(self, val):
        """Update the rotation when slider value changes."""
        self.angle = val
        self.rotated_image = self.rotate_image(self.original_image, self.angle)
        self.rotated_rgb = cv2.cvtColor(self.rotated_image, cv2.COLOR_BGR2RGB)
        self.im2.set_data(self.rotated_rgb)
        self.ax2.set_title(f'Rotated Image ({self.angle:.1f}°)')
        self.fig.canvas.draw_idle()

    def rotate_image(self, image, angle):
        """Rotate image by specified angle in degrees."""
        if angle == 0:
            return image


        height, width = image.shape[:2]


        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)


        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))


        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]


        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                       borderValue=(255, 255, 255))

        return rotated_image

    def save_image(self, event):
        """Save the rotated image."""
        base_name = os.path.splitext(self.image_path)[0]
        extension = os.path.splitext(self.image_path)[1]
        output_path = f"{base_name}_rotated_{self.angle:.1f}deg{extension}"

        cv2.imwrite(output_path, self.rotated_image)
        print(f"Image saved as: {output_path}")

    def show(self):
        """Display the interactive rotation tool."""
        plt.show()

def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python rotate_image_interactive.py <image_path>")
        print("Example: python rotate_image_interactive.py image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)

    try:
        rotator = ImageRotator(image_path)
        rotator.show()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
