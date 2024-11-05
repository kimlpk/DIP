import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('aerial_view.tif', cv2.IMREAD_GRAYSCALE)

# Check if the image is valid
if image is None or not isinstance(image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

# Function to perform histogram equalization
def histogram_equalization(image):

    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    
    return gray_image, equalized_image

# Perform histogram equalization
gray_image, equalized_image = histogram_equalization(image)

cv2.imwrite('aerial_equalized.png', equalized_image)

plt.figure(figsize = (16, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap = 'gray')
plt.axis('off')

# Original Image Histogram
plt.subplot(2, 2, 2)
plt.title('Original Image Histogram')
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.hist(gray_image.ravel(), bins = 256, range = [0, 256], color = 'blue')
plt.xlim([0, 256])

# Equalized Image
plt.subplot(2, 2, 3)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap = 'gray')
plt.axis('off')

# Equalized Image Histogram
plt.subplot(2, 2, 4)
plt.title('Equalized Image Histogram')
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.hist(equalized_image.ravel(), bins = 256, range = [0, 256], color = 'blue')
plt.xlim([0, 256])

# Show plots
plt.tight_layout()
plt.show()