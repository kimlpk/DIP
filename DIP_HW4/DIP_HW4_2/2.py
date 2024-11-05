import cv2
import numpy as np
import matplotlib.pyplot as plt

def notch_reject_filter(shape, d0 = 9, u_k = 0, v_k = 0):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0

    return H

# Read the image
image = cv2.imread('car-moire-pattern.tif', cv2.IMREAD_GRAYSCALE)

# Check if the image is valid
if image is None or not isinstance(image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
phase_spectrumR = np.angle(fshift)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

image_shape = image.shape

H1 = notch_reject_filter(image_shape, 4, 38, 30)
H2 = notch_reject_filter(image_shape, 4, -42, 27)
H3 = notch_reject_filter(image_shape, 2, 80, 30)
H4 = notch_reject_filter(image_shape, 2, -82, 28)

NotchFilter = H1 * H2 * H3 * H4
NotchRejectCenter = fshift * NotchFilter 
NotchReject = np.fft.ifftshift(NotchRejectCenter)
inverse_NotchReject = np.fft.ifft2(NotchReject)  # Compute the inverse DFT of the result

result = np.abs(inverse_NotchReject)

# Save the image
cv2.imwrite('car_original_image.png', image)
cv2.imwrite('car_smagnitude_spectrum_image.png', magnitude_spectrum)
cv2.imwrite('car_notch_filter_image.png', magnitude_spectrum * NotchFilter)
cv2.imwrite('car_filtered_image.png', result)

plt.figure(figsize = (18, 8))

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
plt.title("Notch  Reject Filter in Frequency Domain")
plt.imshow(magnitude_spectrum * NotchFilter, cmap = 'gray') 

# Filtered image
plt.subplot(1, 4, 4)
plt.title("Filtered Image (Notch Filter)")
plt.imshow(result, cmap = 'gray') 

# Show plots
plt.tight_layout()
plt.show()

