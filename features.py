'''
Grab features
'''

import cv2
import numpy as np
import sys

def keypoints_and_descriptors(img):
	gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	sift = cv2.SIFT()
	kp, descriptors = sift.detectAndCompute(gray,None)

	return kp, descriptors

def is_point_in_cell(pt, grid_top_left, height, width):
	x_cond = (pt[0] >= grid_top_left[0]) and (pt[0] < grid_top_left[0] + width)
	y_cond = (pt[1] >= grid_top_left[1]) and (pt[1] < grid_top_left[1] + height)

	return x_cond and y_cond

def get_grid_boundaries(im):
	grid_boundaries = []

	shape = im.shape
	rows = shape[0]
	cols = shape[1]

	for row in np.arange(0, rows, rows / 4.0):
		for col in np.arange(0, cols, cols / 4.0):
			grid_boundaries.append((row, col))

	return grid_boundaries

def fit_keypoints_to_grid(im):
	grid_points = {}
	grid_boundaries = get_grid_boundaries(im)
	grid_width = im.shape[0] / 4.0
	grid_height = im.shape[1] / 4.0
	
	keypoints, descriptors = keypoints_and_descriptors(im)

	kp_desc = zip(keypoints, descriptors)

	for keypoint, descriptor in kp_desc:
		for i, grid in enumerate(grid_boundaries):
			if is_point_in_cell(keypoint.pt, grid, grid_height, grid_width):
				if i in grid_points:
					grid_points[i].append((keypoint, descriptor))
				else:
					grid_points[i] = [(keypoint, descriptor)]

				break
	return grid_points

if __name__ == '__main__':
	img = cv2.imread(sys.argv[1])
	print fit_keypoints_to_grid(img)