import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

from skimage import data
from skimage.util import img_as_ubyte
from skimage import exposure
import skimage.morphology as morp
from skimage.filters import rank

# Constant configuration
k1 = 0
k2 = 0.4
k3 = 0
k4 = 0.05
width = 3

# Estimate of the probability that intensity r
def p(r_counts, total_pixel):
    return r_counts / total_pixel 

# Estimate the mean intensity of a neighbor of pixels
def mean(hist, total_pixel):
    r_values = np.arange(256)
    return np.sum(r_values * hist) / total_pixel

# Estimate the variance of a neighbor of pixels
def var(mean, hist, total_pixel):
    r_values = np.arange(256)
    return np.sum((r_values - mean) ** 2 * hist) / total_pixel


# Estimate the local mean intensity of a neighbor of pixels
# Width is the side length of the neighbor, and it could only be odd numbers
def local_hist(image, x, y, width):
        if width % 2 == 0:
            print("The width should be odd numbers.")
        else:
            offset = int((width - 1) / 2)
            local_region = image[y - offset : y + offset + 1, x - offset : x + offset + 1]
            hist = cv2.calcHist([local_region], [0], None, [256], [0, 256])
            hist = hist.reshape((256, ))
            return hist
        
def hist_value(hist):
    min, max = 0, 0
    for i in range(0, 255):
        if hist[i] != 0:
            if(i>max):
                max = i
            elif(i < min):
                min = i
            else:
                pass
    return [min, max]

# Read the image
image = cv2.imread('hidden_object_2.jpg', cv2.IMREAD_GRAYSCALE)
new_image = image.copy()

# Check if the image is valid
if image is None or not isinstance(image, (np.ndarray)):
    raise ValueError("Invalid image provided. Please check the file path.")

# Calculate the histogram without normalization (count)
global_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
global_hist = global_hist.reshape((256, ))

# Calulate the global mean intensity
shape_x = np.shape(image)[1]
shape_y = np.shape(image)[0]
total_pixel = shape_x * shape_y
mean_global = mean(global_hist, total_pixel)
print("---------")
print("Global mean:", mean_global)
deviation_global = math.sqrt(var(mean_global, global_hist, total_pixel))
print("---------")
print("Global deviation:", deviation_global)
print("---------")
print("width:", width)
print("---------")
print("Offset:", int((width - 1) / 2))
print("---------")
# c = hist_value(global_hist)[1] / (mean_global * k2)
c = 25
print("---------")
print("c:", c)

for pixel_y in range(0, shape_y):
    processed_count = 0
    for pixel_x in range(0, shape_x):
        offset = int((width - 1) / 2)
        local_pixel = width ** 2
        if(pixel_y - offset < 0 or pixel_x - offset < 0):
            pass
        elif(pixel_y + offset > shape_y or pixel_x + offset > shape_x):
            pass
        else:
            hist = local_hist(image, pixel_x, pixel_y, width)
            mean_local = mean(hist, local_pixel)
            deviation_local = math.sqrt(var(mean_local, hist, local_pixel))
            # print("-----------------------")
            # print("deal with [%d][%d]" % (pixel_y, pixel_x))
            # print("mean local:",mean_local," deviation local:", deviation_local)
            if((k1*mean_global <= mean_local and mean_local <= k2*mean_global) and 
               (k3*deviation_global <= deviation_local and deviation_local <= k4*deviation_global)):
                try:
                    # new_image[pixel_y][pixel_x] *= c
                    new_image[pixel_y - offset : pixel_y + offset + 1, pixel_x - offset : pixel_x + offset + 1] *= c
                    # new_image[pixel_y - 1 : pixel_y+1 + 1, pixel_x - 1:pixel_x+1 + 1] *= c

                    processed_count += 1
                except:
                    print("Index out of bounds")
            else:
                pass
    print("finish dealing with with %dth row and %d pixels are processed" % (pixel_y,processed_count)) 
        
# Show the original image and the processed image
plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap = 'gray')
plt.subplot(1, 2, 2)
plt.title("New Image")
plt.imshow(new_image, cmap = 'gray')

# Show the original histogram and the processed histogram
plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
x = np.arange(256)
plt.bar(x, global_hist, width = 1.0, color = "blue") 
plt.xlim([0, 256])

processed_hist = cv2.calcHist([new_image], [0], None, [256], [0, 256])
processed_hist = processed_hist.reshape((256, ))

plt.subplot(1, 2, 2)
plt.title("Processed Histogram")
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")
x = np.arange(256)
plt.bar(x, processed_hist, width = 1.0, color = "blue") 
plt.xlim([0, 256])

# Show plots
plt.show()

