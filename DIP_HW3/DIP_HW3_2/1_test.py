import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('hidden_object_2.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is valid
if image is None or not isinstance(image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

# Step 1: Histogram Statistics Method
# Calculate histogram
hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# Calculate the cumulative distribution function (CDF)
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Use Otsu's method to find a suitable threshold
thresh_val, thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show the threshold image
plt.figure(figsize = (12, 6))
plt.title('Thresholded Image')
plt.imshow(thresh_img, cmap = 'gray')
plt.show()

# Step 2: Local Enhancement Method
# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE()
enhanced_img = clahe.apply(image)

# Show the enhanced image
plt.figure(figsize = (12, 6))
plt.title('Enhanced Image')
plt.imshow(enhanced_img, cmap = 'gray')
plt.show()

# Save the results
cv2.imwrite('thresholded_image.jpg', thresh_img)
cv2.imwrite('enhanced_image.jpg', enhanced_img)

# Show plots
plt.show()

