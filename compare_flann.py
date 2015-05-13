'''
Use traditional alignment
'''

import numpy as np
import sys
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine, euclidean

SIFT = cv2.SIFT()

def get_sift_descriptors_and_matches(img1, img2):
	'''
	img1 is test image, img2 is template
	'''
	kp1, des1 = SIFT.detectAndCompute(img1, None)
	kp2, des2 = SIFT.detectAndCompute(img2, None)

	# FLANN parameters
	detector = cv2.FeatureDetector_create("SIFT")
	descriptor = cv2.DescriptorExtractor_create("SIFT")

	kps1 = detector.detect(img1)
	kps1, descr1 = descriptor.compute(img1, kps1)

	kps2 = detector.detect(img2)
	kps2, descr2 = descriptor.compute(img2, kps2)

	flann_params = dict(algorithm=1, trees=10)
	flann = cv2.flann_Index(descr1, flann_params)
	idx, dist = flann.knnSearch(descr2, 1, params={})

	return idx, dist, descr1, descr2

def image_distance_inner(img1, img2):
	im1 = cv2.imread(img1)
	im2 = cv2.imread(img2)
	idx, dist, descr1, descr2 = get_sift_descriptors_and_matches(im1, im2)

	idx = idx.flatten()
	dist = dist.flatten()
	distance = 0.0
	for i, src_i in enumerate(idx.flatten()):
		template_descriptor = descr2[i]
		src_descriptor = descr1[int(src_i)]

		distance += euclidean(src_descriptor, template_descriptor)
		#distance += dist[i]
	return distance
	#return np.average(dist)

def image_distance(img1, img2):
	return min([image_distance_inner(img1, img2), image_distance_inner(img2, img1)])

if __name__ == '__main__':
	img1, img2 = sys.argv[1:]
	print image_distance(img1, img2)
