import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('astronaut-interference.tif', cv2.IMREAD_GRAYSCALE)

# Check if the image is valid
if image is None or not isinstance(image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

# Perform FFT to transform the image into the frequency domain
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

# Create a 2D notch filter to remove specific frequencies
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
D0 = 35  # Cutoff frequency for the bandstop filter
W = 11  # Width of the notch filter

notch_filter = np.ones((rows, cols), np.uint8)

# Create notches around specific frequencies
notch_filter[crow - D0 - W : crow - D0 + W, ccol - D0 - W : ccol - D0 + W] = 0
notch_filter[crow + D0 - W : crow + D0 + W, ccol + D0 - W : ccol + D0 + W] = 0

# Apply the notch filter in the frequency domain
filtered_f_transform_shifted = f_transform_shifted * notch_filter

# Inverse transform to the spatial domain
filtered_image = np.fft.ifftshift(filtered_f_transform_shifted)
filtered_image = np.fft.ifft2(filtered_image)
filtered_image = np.abs(filtered_image).astype(np.uint8)

# Calculate the magnitude spectrum of the filtered image
filtered_magnitude_spectrum = np.log(np.abs(filtered_f_transform_shifted) + 1)

# Save the image
cv2.imwrite('astronaut_original_image.png', image)
cv2.imwrite('astronaute_smagnitude_spectrum_image.png', magnitude_spectrum)
cv2.imwrite('astronaut_notch_filter_image.png', notch_filter)
cv2.imwrite('astronaut_filtered_image.png', filtered_image)

plt.figure(figsize = (18, 6))

# Display the original image
plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image, cmap = 'gray')

# Frequency domain
plt.subplot(1, 4, 2)
plt.title("Frequency Domain Image")
plt.imshow(magnitude_spectrum, cmap = 'gray')

# Filter
plt.subplot(1, 4, 3)
plt.title("Notch Filter in Frequency Domain")
plt.imshow(notch_filter, cmap = 'gray')

# Filtered image
plt.subplot(1, 4, 4)
plt.title("Filtered Image (Notch Filter)")
plt.imshow(filtered_image, cmap = 'gray')

# Show plots
plt.tight_layout()
plt.show()

