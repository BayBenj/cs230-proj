#!/usr/bin/env python3

import numpy as np
import cv2

# Read the images to be aligned
#First image is output image Y
im1 =  cv2.imread("/proj/data/fchdr_rend_dataset/DelicateArchRendered.jpg");

#Second is input image X (This will be aligned)
im2 =  cv2.imread("/proj/data/fchdr_rgb_dataset/s020_delicate-arch/s020_mdf0124.png");

size_req = (im1.shape[1], im1.shape[0])
#im2 = cv2.resize(im2, dsize=size_req, interpolation=cv2.INTER_CUBIC)

#print(im1)
# Convert images to grayscale
im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
 
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
