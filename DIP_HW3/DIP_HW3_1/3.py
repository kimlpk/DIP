import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
original_image = cv2.imread('aerial_view.tif', cv2.IMREAD_GRAYSCALE)

# Check if the image is valid
if original_image is None or not isinstance(original_image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

# Define the target probability density function p(zq)
def target_pdf(zq):
    return zq ** 0.4

# Calculate the total probability by numerically integrating the target PDF
total_probability = 0
for zq in range(256):
    total_probability += target_pdf(zq)

# Normalize the total probability to obtain a normalized PDF
normalized_pdf = [target_pdf(zq) / total_probability for zq in range(256)]

# Calculate the value of c as the reciprocal of the total probability
c = 1 / total_probability

# Calculate the original histogram
original_histogram = cv2.calcHist([original_image], [0], None, [256], [0, 256])

# Normalize the original histogram to get the original probability density function p(z)
original_pdf = original_histogram / original_histogram.sum()

# Calculate the cumulative distribution functions (CDFs) of the original and target PDFs
original_cdf = np.cumsum(original_pdf)
target_cdf_values = [c * (zq ** 0.4) for zq in range(256)]
target_cdf = np.cumsum(target_cdf_values)

# Calculate the mapping function
mapping_function = np.zeros(256, dtype = np.uint8)
for z in range(256):
    mapping_function[z] = np.argmin(np.abs(target_cdf - original_cdf[z]))

# Apply the mapping function to the original image
matched_image = mapping_function[original_image]

cv2.imwrite('matched.png', matched_image)

plt.figure(figsize = (16, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(original_image, cmap = 'gray')
plt.axis('off')

# Original Image Histogram
plt.subplot(2, 2, 2)
plt.title('Original Image Histogram')
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.hist(original_image.ravel(), bins = 256, range = [0, 256], color = 'blue')
plt.xlim([0, 256])

# Matched Image
plt.subplot(2, 2, 3)
plt.title('Matched Image')
plt.imshow(matched_image, cmap = 'gray')
plt.axis('off')

# Matched Image Histogram
plt.subplot(2, 2, 4)
plt.title('Matched Image Histogram')
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
plt.hist(matched_image.ravel(), bins = 256, range = [0, 256], color = 'blue')
plt.xlim([0, 256])

plt.figure(figsize = (16, 8))

# Plot the target PDF (probability density function)
plt.title("Target PDF")
plt.xlabel("Intensity")
plt.ylabel("Probability")
plt.plot(target_cdf_values)
plt.xlim([0, 256])

# Show plots
plt.tight_layout()
plt.show()

# Print the calculated value of c
print("Calculated value of c:", c)