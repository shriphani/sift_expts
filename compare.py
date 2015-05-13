'''
Distance metrics between two images
'''

from scipy.spatial.distance import cosine, euclidean
import cv2
import numpy as np
import sys
import features

def compute_min_distance(kps1, kps2, distance_metric = cosine):
	'''
	Given two keypoints, compute the minimum distances
	'''

	dist = float('inf')

	for kp1 in kps1:
		for kp2 in kps2:
			v1 = kp1[1]
			v2 = kp2[1]

			if distance_metric(v1, v2) < dist:
				dist = distance_metric(v1, v2)

	return dist

def image_distance(img1, img2, num_cells, dist = cosine):
	im1 = cv2.imread(img1)
	im2 = cv2.imread(img2)

	feats1 = features.fit_keypoints_to_grid(im1)
	feats2 = features.fit_keypoints_to_grid(im2)

	grid_dists = []

	for i in range(num_cells):
		if i in feats1 and i in feats2:
			kps_descs1 = feats1[i]
			kps_descs2 = feats2[i]

			grid_dists.append(compute_min_distance(kps_descs1, kps_descs2, distance_metric = dist))

	return np.average(np.array(grid_dists))

if __name__ == '__main__':
	img1, img2, num_cells = sys.argv[1:]
	num_cells = int(num_cells)
	print image_distance(img1, img2, num_cells)