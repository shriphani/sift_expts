'''
Given an lfw dataset, compare images
'''
from compare import image_distance
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
import numpy as np
import sys

def get_dists(handle, dist):
	for new_line in handle:
		img1, img2 = new_line.strip().split()

		img1_file = '/Users/shriphani/Downloads/lfwcrop_color/faces/' + img1 + '.ppm'
		img2_file = '/Users/shriphani/Downloads/lfwcrop_color/faces/' + img2 + '.ppm'

		yield image_distance(img1_file, img2_file, 16, dist)

if __name__ == '__main__':
	same_tests, diff_tests = sys.argv[1:]

	with open(same_tests, 'r') as same_handle, open(diff_tests, 'r') as diff_handle:
		same_dists_cosine = list(get_dists(same_handle, cosine))
		diff_dists_cosine = list(get_dists(diff_handle, cosine))
		
		plt.hist(same_dists_cosine, color='green')
		plt.hist(diff_dists_cosine, color='red')

		plt.savefig('sift_cosine.png')

		plt.clf()

	with open(same_tests, 'r') as same_handle, open(diff_tests, 'r') as diff_handle:
		same_dists_euclidean = np.array(list(get_dists(same_handle, euclidean)))
		diff_dists_euclidean = np.array(list(get_dists(diff_handle, euclidean)))
		
		plt.hist(same_dists_euclidean, color='green')
		plt.hist(diff_dists_euclidean, color='red')

		plt.savefig('sift_euclidean.png')
