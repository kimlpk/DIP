import cv2
import numpy as np
import matplotlib.pyplot as plt

k0 = 0
k1 = 0.3
k2 = 0
k3 = 0.05

def statistical_measures(image, kernal_size):
    if kernal_size == 0:
        return cv2.meanStdDev(image)[0], cv2.meanStdDev(image)[1]
    else:
        i = kernal_size
        kernal = np.ones((i, i), np.float32) / (i ** 2)
        mean = cv2.filter2D(image.astype(np.float32), -1, kernal)
        variance = cv2.filter2D((image.astype(np.float32) - mean) ** 2, -1, kernal)
        return mean, np.sqrt(variance)

def boundary(a, b):
    a_lower_bound = k0 * a
    a_upper_bound = k1 * a
    b_lower_bound = k2 * b
    b_upper_bound = k3 * b
    return [a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound]

def histogram_statistics(image, local_ms, max, b):
    width, height = image.shape

    for i in range(1, width):
        for j in range(1, height):
            local = image[i - 1 : i + 2, j - 1 : j + 2]
            local_max = np.max(local)
            if (b[0] < local_ms[0][i, j] < b[1]) and (b[2] < local_ms[1][i - 1, j - 1] < b[3]):
                c = round(max / local_max)
                image[i, j] = round(c * image[i, j])
    return image

# Read the image
image = cv2.imread('hidden_object_2.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is valid
if image is None or not isinstance(image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

local = statistical_measures(image, 3)
max = np.max(image)
mean_std = statistical_measures(image, 0)

# histogram_statistics method
b = boundary(mean_std[0], mean_std[1])
image = histogram_statistics(image, local, max, b)
cv2.imwrite('histogram_statistics_image.png', image)

# local enhancement method
clahe = cv2.createCLAHE()
clahe_image = clahe.apply(image)
cv2.imwrite('local_enhancement_image.png', clahe_image)

# Show plots
plt.show()

