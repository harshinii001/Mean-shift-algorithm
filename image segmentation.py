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

            c = current_point * kernel[:, np.newaxis]

            yi_new = np.sum(c, axis=0) / np.sum(kernel)

            if np.linalg.norm(yi_new - yi) < convergence_threshold:
                yi_conv = yi_new
                converged = True
            else:
                yi = yi_new

        filtered_points[i] = yi_conv
        i += 1

    # Perform cluster delineation
    clusters = []
    visited = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        if visited[i]:
            continue

        cluster = [i]
        visited[i] = True

        for j in range(i+1, len(points)):
            if not visited[j]:
                if np.linalg.norm(filtered_points[i, :2] - filtered_points[j, :2]) <= h_s and np.linalg.norm(filtered_points[i, 2:] - filtered_points[j, 2:]) <= h_r:
                    cluster.append(j)
                    visited[j] = True

        clusters.append(cluster)

    # Perform image segmentation
    min_region_size = 300  # Minimum number of pixels for a region
    labels = np.zeros(len(points), dtype=int)
    segment_id = 1

    for i, cluster in enumerate(clusters):
        if len(cluster) >= min_region_size:
            for idx in cluster:
                labels[idx] = segment_id

            segment_id += 1

    segmented_image = labels.reshape((height, width))

    # Create color map
    color_map = {}
    for i, cluster in enumerate(clusters):
        if len(cluster) >= min_region_size:
            color_map[i + 1] = np.random.randint(0, 255, size=3)  # Assign a random RGB color

    # Assign colors to each pixel based on cluster labels
    segmented_image_rgb = np.zeros_like(image)
    for i, cluster in enumerate(clusters):
        if len(cluster) >= min_region_size:
            color = color_map[i + 1]
            for idx in cluster:
                x, y = int(points[idx, 0]), int(points[idx, 1])
                segmented_image_rgb[x, y] = color

    return segmented_image_rgb

def main():
    # Load the input image
    image_path = 'image6.png'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Define the bandwidths for spatial and range domains
    h_s = 130.0  # Increase the spatial bandwidth
    h_r = 25.0  # Increase the range bandwidth

    # Define the truncation value for the Gaussian distribution
    truncation = 400  # Increase the truncation value

    # Perform mean shift image smoothing and segmentation
    segmented_image = mean_shift_smoothing(image, h_s, h_r, truncation)

    # Display the original and segmented images
    # cv2.imshow('Original Image', image)
    cv2.imwrite('Segmented_Image_5.jpg', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
