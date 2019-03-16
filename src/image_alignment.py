#!/usr/bin/env python3

import numpy as np
import cv2

# Read the images to be aligned
#First image is output image Y
im1 =  cv2.imread("/proj/data/fchdr_rend_dataset/DelicateArchRendered.jpg");

#Second is input image X (This will be aligned)
im2 =  cv2.imread("/proj/data/fchdr_rgb_dataset/s020_delicate-arch/s020_mdf0120.png");
cv2.imwrite("./output/x_inp_noresize.jpg", im2)
#im3 =  cv2.imread("/proj/data/fchdr_rgb_dataset/s020_delicate-arch/s020_mdf0121.png");
#im4 =  cv2.imread("/proj/data/fchdr_rgb_dataset/s020_delicate-arch/s020_mdf0122.png");
#im5 =  cv2.imread("/proj/data/fchdr_rgb_dataset/s020_delicate-arch/s020_mdf0123.png");
#im6 =  cv2.imread("/proj/data/fchdr_rgb_dataset/s020_delicate-arch/s020_mdf0124.png");
#im7 =  cv2.imread("/proj/data/fchdr_rgb_dataset/s020_delicate-arch/s020_mdf0125.png");
#im8 =  cv2.imread("/proj/data/fchdr_rgb_dataset/s020_delicate-arch/s020_mdf0126.png");
#im9 =  cv2.imread("/proj/data/fchdr_rgb_dataset/s020_delicate-arch/s020_mdf0127.png");
#im10 =  cv2.imread("/proj/data/fchdr_rgb_dataset/s020_delicate-arch/s020_mdf0128.png");

size_req = (im1.shape[1], im1.shape[0])
im2 = cv2.resize(im2, dsize=size_req, interpolation=cv2.INTER_NEAREST)
#im3 = cv2.resize(im3, dsize=size_req, interpolation=cv2.INTER_CUBIC)
#im4 = cv2.resize(im4, dsize=size_req, interpolation=cv2.INTER_CUBIC)
#im5 = cv2.resize(im5, dsize=size_req, interpolation=cv2.INTER_CUBIC)
#im6 = cv2.resize(im6, dsize=size_req, interpolation=cv2.INTER_CUBIC)
#im7 = cv2.resize(im7, dsize=size_req, interpolation=cv2.INTER_CUBIC)
#im8 = cv2.resize(im8, dsize=size_req, interpolation=cv2.INTER_CUBIC)
#im9 = cv2.resize(im9, dsize=size_req, interpolation=cv2.INTER_CUBIC)
#im10 = cv2.resize(im10, dsize=size_req, interpolation=cv2.INTER_CUBIC)
#
#print(im1)
# Convert images to grayscale
im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
#im3_gray = cv2.cvtColor(im3, cv2.COLOR_RGB2GRAY)
#im4_gray = cv2.cvtColor(im4, cv2.COLOR_RGB2GRAY)
#im5_gray = cv2.cvtColor(im5, cv2.COLOR_RGB2GRAY)
#im6_gray = cv2.cvtColor(im6, cv2.COLOR_RGB2GRAY)
#im7_gray = cv2.cvtColor(im7, cv2.COLOR_RGB2GRAY)
#im8_gray = cv2.cvtColor(im8, cv2.COLOR_RGB2GRAY)
#im9_gray = cv2.cvtColor(im9, cv2.COLOR_RGB2GRAY)
#im10_gray = cv2.cvtColor(im10, cv2.COLOR_RGB2GRAY)
 
# Find size of image1
sz = im1.shape
 
# Define the motion model
warp_mode = cv2.MOTION_AFFINE
 
# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
 
# Specify the number of iterations.
number_of_iterations = 5000;
 
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;
 
# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
# Run the ECC algorithm. The results are stored in warp_matrix.
#(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im3_gray,warp_matrix, warp_mode, criteria)
#print(warp_matrix)
#(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im4_gray,warp_matrix, warp_mode, criteria)
#print(warp_matrix)
#(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im5_gray,warp_matrix, warp_mode, criteria)
#print(warp_matrix)
#(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im6_gray,warp_matrix, warp_mode, criteria)
#print(warp_matrix)
#(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im7_gray,warp_matrix, warp_mode, criteria)
#print(warp_matrix)
#(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im8_gray,warp_matrix, warp_mode, criteria)
#print(warp_matrix)
#(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im9_gray,warp_matrix, warp_mode, criteria)
#print(warp_matrix)
#(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im10_gray,warp_matrix, warp_mode, criteria)
#print(warp_matrix)
(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
print(warp_matrix)

if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
    im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else :
        # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
 
# Show final results
cv2.imwrite("./output/y.jpg", im1)
cv2.imwrite("./output/x_inp.jpg", im2)
cv2.imwrite("./output/x_aligned.jpg", im2_aligned)
