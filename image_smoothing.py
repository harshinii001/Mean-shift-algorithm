#!/usr/bin/env python3
import numpy as np
import cv2

def truncated_gaussian(x, sigma, truncation):
    # Truncated Gaussian function
    return np.exp(-(x ** 2) / (2 * sigma ** 2)) * (np.abs(x) <= truncation)

def mean_shift_smoothing(image, h_s, h_r, truncation, convergence_threshold=1):
    luv_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    height, width, _ = luv_image.shape

    # Create an array to store the points
    points = np.zeros((height * width, 5))
    idx = 0

    # Iterate over each pixel and store x, y, L, u, v values
    for i in range(height):
        for j in range(width):
            points[idx] = [i, j, luv_image[i, j, 0], luv_image[i, j, 1], luv_image[i, j, 2]]
            idx += 1

    filtered_points = np.zeros_like(points)

    i = 0
    while i < len(points):
        current_point = points[i]

        yi = np.copy(current_point)
        converged = False

        while not converged:
            kernel_spatial = truncated_gaussian(np.linalg.norm(points[:, :2] - yi[:2], axis=1), h_s, truncation)
            kernel_range = truncated_gaussian(np.linalg.norm(points[:, 2:] - yi[2:], axis=1), h_r, truncation)
            kernel = kernel_spatial * kernel_range

            c = points * kernel[:, np.newaxis]

            yi_new = np.sum(c, axis=0) / np.sum(kernel)

            if np.linalg.norm(yi_new - yi) < convergence_threshold:
                yi_conv = yi_new
                converged = True
            else:
                yi = yi_new

        filtered_points[i] = yi_conv
        i += 1

    filtered_image_luv = filtered_points[:, 2:].reshape((height, width, 3))
    filtered_image_luv = np.clip(filtered_image_luv, 0, 255).astype(np.uint8)
    filtered_image_rgb = cv2.cvtColor(filtered_image_luv, cv2.COLOR_LUV2RGB)

    return filtered_image_rgb

def main():
    # Load the input image
    image_path = 'image6.png'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Define the bandwidths for spatial and range domains
    h_s = 80.0
    h_r = 80.0

    # Define the truncation value for the Gaussian distribution
    truncation = 30

    # Perform mean shift image smoothing
    filtered_image = mean_shift_smoothing(image, h_s, h_r, truncation)

    # Display the original and filtered images
    #cv2.imshow('Original Image', image)
    cv2.imwrite('Filtered_Image_3.png', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
