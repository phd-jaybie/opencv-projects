# Name: Always on 2D object detection 
# Author: Jaybie A. de Guzman, 		Date: 20-Sep-2017
# Description:
# Adapted from the image detection sample code from the openc-python
# tutorial at http://docs.opencv.org/3.3.0/d1/de0/tutorial_py_feature_homography.html

import numpy as np
import cv2
#from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('train.jpg',0)          # queryImage
#img2 = cv2.imread('box_in_scene.png',0) # trainImage
cap = cv2.VideoCapture(0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors from the training image with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)

while(1):
	# get each frame
	_, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# find features on the frame with SIFT
	kp2, des2 = sift.detectAndCompute(frame,None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
	
	# if enough matches are found
	if len(good)>MIN_MATCH_COUNT:
		# extract location of points in both images
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		# find the perspective transform
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()
		
		# get the transform points in the (captured) query image
		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		# draw the transformed image
		res = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None

	#draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)

	#img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

	cv2.imshow('Tracking', res)
	k = cv2.waitKey(5) & 0xFF
	if k ==27:
		break

cap.release()
cv2.destroyAllWindows()
