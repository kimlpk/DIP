import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    """Generate a Gaussian filter kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    kernel = kernel / np.sum(kernel)  # Normalize
    return kernel

kernel_size = 201  # Size of the kernel
sigma_value = 45  # Standard deviation

# Generate Gaussian kern
gaussian_kernel_result = gaussian_kernel(kernel_size, sigma_value)

# Read the image
image = cv2.imread('N1.bmp', cv2.IMREAD_GRAYSCALE)

# Check if the image is valid
if image is None or not isinstance(image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

cvfilter = cv2.filter2D(image, -1, gaussian_kernel_result)
image2 = image / cvfilter

# Save the image
cv2.imwrite('N1_original.png', image)
cv2.imwrite('N1_shaded.png', cvfilter)
cv2.imwrite('N1_gaussian.png', image2)

plt.figure(figsize = (12, 6))

# Show the original image and the processed image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap = 'gray')

plt.subplot(1, 3, 2)
plt.title("Shaded Pattern")
plt.imshow(cvfilter, cmap = 'gray')

plt.subplot(1, 3, 3)
plt.title("Shaded Corrected Image")
plt.imshow(image2, cmap = 'gray')

# Show plots
plt.show()

