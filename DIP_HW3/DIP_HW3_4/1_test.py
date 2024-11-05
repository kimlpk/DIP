import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('Bodybone.bmp', cv2.IMREAD_GRAYSCALE)

# Check if the image is valid
if image is None or not isinstance(image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

# Define Sobel filter kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

# Define Laplacian filter kernel
laplacian = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])

# Initialize filtered images
sobel_magnitude = np.zeros_like(image)
laplacian_output = np.zeros_like(image)

# Perform Sobel filtering
rows, cols = image.shape
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        # Sobel X
        gradient_x = (sobel_x * image[i - 1 : i + 2, j - 1 : j + 2]).sum()
        # Sobel Y
        gradient_y = (sobel_y * image[i - 1 : i + 2, j - 1 : j + 2]).sum()
        sobel_magnitude[i, j] = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

# Perform Laplacian filtering
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        laplacian_output[i, j] = (laplacian * image[i - 1 : i + 2, j - 1 : j + 2]).sum()
laplacian_image = np.uint8(np.abs(laplacian_output))

# Set high boost factor
alpha = 1.5

# High boost filtering
highboost = cv2.addWeighted(image, 1 + alpha, np.clip(np.abs(laplacian_output), 0, 255), -alpha, 0)

plt.figure(figsize = (12, 10))

# Display results
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap = 'gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Sobel Filtered Image')
plt.imshow(sobel_magnitude, cmap = 'gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Laplacian Filtered Image')
plt.imshow(laplacian_image, cmap = 'gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('High-Boost Filtered Image')
plt.imshow(highboost, cmap = 'gray')
plt.axis('off')

# Show plots
plt.tight_layout()
plt.show()


