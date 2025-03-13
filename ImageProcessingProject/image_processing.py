import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("c:/Users/Admin/Desktop/session1/ImageProcessingProject/images/sample.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# Display the image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()