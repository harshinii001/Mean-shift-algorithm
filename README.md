# Implementation of Mean-Shift Image Smoothing with Edge Preservation and Mean-Shift Image Segmentation from Scratch

This project provides the algorithm based on the research paper "Mean Shift: A Robust Approach Toward Feature Space Analysis". The implementation uses truncated Gaussian distribution for the spatial kernel and range spatial kernel of the multivariable kernel density estimator.

The smoothness of the image can be adjusted by changing the hs and hr bandwidth parameters of the spatial kernel and range spatial kernel. Similarly, the segmentation of the image can be customized by adjusting the hs and hr bandwidth parameters of the spatial kernel and range spatial kernel, as well as the minimum cluster size. These parameters can be fine-tuned through trial and error to achieve the desired result.

#Mean-Shift Image Smoothing with Edge Preservation

The algorithm converts the input image to the LUV color space and uses spatial (for smoothing) and range and space (for edge preservation) kernels to determine the color intensity of the pixels. In Mean Shift iterations, data points move toward their respective mode until each pixel converges. Unlike K-means clustering, the number of clusters does not need to be specified. Depending on the kernel size and type, clustering will be affected.

#Mean-Shift Image Segmentation
This code extends the image smoothing algorithm using Mean Shift by incorporating cluster delineation and image segmentation. It identifies clusters of pixels based on their spatial and range similarities after the smoothing process. Then, it assigns unique colors to each cluster to visually represent the segmented regions in the image.

These algorithms have a time complexity of O(N^2). It is recommended to use small-sized input images for better performance.

