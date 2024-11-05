import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('aerial_view.tif', cv2.IMREAD_GRAYSCALE)

# Check if the image is valid
if image is None or not isinstance(image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

# Function to perform histogram
def histogram(image):

    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    return gray_image

# Perform histogram
gray_image = histogram(image)

cv2.imwrite('aerial_view.png', gray_image)

plt.figure(figsize = (16, 8))

# Image
plt.subplot(1, 2, 1)
plt.title('Aerial View Image')
plt.imshow(image, cmap = 'gray')
plt.axis('off')

# Histogram
plt.subplot(1, 2, 2)
plt.title('Histogram')
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.hist(gray_image.ravel(), bins = 256, range = [0, 256], color = 'blue')
plt.xlim([0, 256])

# Show plots
plt.tight_layout()
plt.show()