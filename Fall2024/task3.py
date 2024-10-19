import cv2
import numpy as np
from scipy import ndimage
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define kernels
gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1,-2,-1]])
emboss_kernel = np.array([[2, 0 ,0], [0,-1,-1], [0,-1,-1]])
outline_kernel_3x3 = np.array([[0,-1 ,0], [-1 ,4 ,-1], [0 ,-1 ,0]])
outline_kernel_5x5 = np.array([[0,-1 ,0 ,2 ,0], [-1 ,4 ,-1 ,0 ,2], [0 ,-1 ,0 ,-1 ,-4]])

# Set up function to run the convolution on an image
def concolve2d(image,kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    padded_image = np.pad(image,
                          ((padding_height,padding_height),
                           (padding_width,padding_width)),
                          mode='constant')

    output = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            output[i,j] = np.sum(padded_image[i:i+kernel_height,
                                               j:j+kernel_width] * kernel)

    return output

# Read and process the image
image_path = '/home/pswish/Downloads/wine.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found or unable to read.")

#Pick two kernels to compare
result_gaussian = ndimage.convolve(image, gaussian_kernel)
result_outline = concolve2d(image, outline_kernel_3x3)

# Display results
plt.figure(figsize=(10,5))

# Plot the results to compare visually
plt.subplot(1,2,1)
plt.imshow(result_gaussian,cmap='gray')
plt.title('Gaussian Blur')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(result_outline,cmap='gray')
plt.title('Outline Effect')
plt.axis('off')

# Save the output to a file
plt.savefig('output_plot.png')

