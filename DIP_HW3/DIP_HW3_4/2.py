import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_kernel():
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    kernel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    return kernel_x, kernel_y

def laplacian_kernel():
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return kernel

def kernel_filter(image, kernel, image_pad):
    image_filter = cv2.filter2D(image_pad, -1, kernel)
    image_filter = image_filter[kernel.shape[0] // 2 : kernel.shape[0] // 2 + image.shape[0], kernel.shape[1] // 2 : kernel.shape[1] // 2 + image.shape[1]]
    return image_filter

def filter(image, kernel):
    kernel_size = kernel.shape[0]
    image_pad = cv2.copyMakeBorder(image, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, cv2.BORDER_REPLICATE)
    filtered_image = kernel_filter(image, kernel, image_pad)
    return filtered_image

def sobel(image):
    kernel_x, kernel_y = sobel_kernel()
    sobel_x = filter(image, kernel_x)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = filter(image, kernel_y)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    
    sobel_xy =  cv2.add(sobel_x, sobel_y)
    sobel_xy = cv2.normalize(sobel_xy, None, 0, 255, cv2.NORM_MINMAX)
    return sobel_xy

def laplacian(image):
    kernel = laplacian_kernel()
    laplacian_image = filter(image, kernel)
    laplacian_image = cv2.normalize(laplacian_image, None, 0, 255, cv2.NORM_MINMAX)
    return laplacian_image
          
# Read the image
image = cv2.imread('fish.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is valid
if image is None or not isinstance(image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

# Sobel
sobel_xy = sobel(image)

# Laplacian
laplacian_image = laplacian(image)

# Combine two filters
# Highboost filtering
result = cv2.addWeighted(sobel_xy, 1, laplacian_image, 1, 0)
result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

# Save the image
cv2.imwrite('fish_original_image.png', image)
cv2.imwrite('fish_sobel_image.png', sobel_xy)
cv2.imwrite('fish_laplacian_image.png', laplacian_image)
cv2.imwrite('fish_final_image.png', result)

plt.figure(figsize = (12, 6))

# Show the original image and the processed image
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap = 'gray')

plt.subplot(2, 2, 2)
plt.title("Sobel Filtered Image")
plt.imshow(sobel_xy, cmap = 'gray')

plt.subplot(2, 2, 3)
plt.title("Laplacian Filtered Image")
plt.imshow(laplacian_image, cmap = 'gray')

plt.subplot(2, 2, 4)
plt.title("High-Boost Filtered Image")
plt.imshow(result, cmap = 'gray')

# Show plots
plt.tight_layout()
plt.show()

