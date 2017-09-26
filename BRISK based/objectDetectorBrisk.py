# Name: Always on 2D object detection 
# Author: Jaybie A. de Guzman, 		Date: 20-Sep-2017
# Description:
# Adapted from the image detection sample code from the openc-python
# tutorial at http://docs.opencv.org/3.3.0/d1/de0/tutorial_py_feature_homography.html

import numpy as np
import cv2
import time
#from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH,\
	table_number = 6, key_size = 12, multi_proble_level = 1)
search_params = dict(checks = 20)

flann = cv2.FlannBasedMatcher(index_params, search_params)

bf = cv2.BFMatcher()#cv2.NORM_HAMMING, crossCheck = True)

img1 = cv2.imread('train.jpg',0)          # queryImage

cap = cv2.VideoCapture(0)

frameCount = 0
# Initiate ORB detector
brisk = cv2.BRISK_create()

# find the keypoints and descriptors from the training image with SIFT
kp1, des1 = brisk.detectAndCompute(img1,None)

print("Frame, Time to process, Training Descriptors, Query Descriptors, Good matches")

while(1):
	# get each frame
	_, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ox,oy = frame.shape
	frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
	rx,ry = frame.shape

	t1 = time.time()
	frameCount = frameCount + 1

	# find features on the frame with SIFT/ORB
	kp2, des2 = brisk.detectAndCompute(frame,None)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
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
		res = frame
		matchesMask = None
	
	t2 = time.time()
	print("%d, %.5f, %d, %d, %d" % (frameCount,t2-t1,len(des1),len(des2),len(good)))

	# get processing time for each frame
	#td = t2 - t1
	#textTime = "Time to process this frame: " + str(td) + " seconds."
	#oSizeInfo = "Original res " + str(ox) + "x" + str(oy)
	#rSizeInfo = "New res " + str(rx) + "x" + str(ry)
	#res = cv2.putText(res,textTime,(10,30), cv2.FONT_HERSHEY_SIMPLEX,\
	#	0.5, (10,10,10),1, cv2.LINE_AA)
	#res = cv2.putText(res,oSizeInfo,(10,45), cv2.FONT_HERSHEY_SIMPLEX,\
	#	0.5, (10,10,10),1, cv2.LINE_AA)
	#res = cv2.putText(res,rSizeInfo,(10,60), cv2.FONT_HERSHEY_SIMPLEX,\
	#	0.5, (10,10,10),1, cv2.LINE_AA)
 
	draw_params = dict(matchColor = (0,255,0), singlePointColor = None,\
		matchesMask = matchesMask, flags = 2)
	img3 = cv2.drawMatches(img1,kp1,frame,kp2,good,None,**draw_params)
	
	#cv2.imshow('Detected', res)
	cv2.imshow('Matches',img3) # shows the matching lines for checking

	k = cv2.waitKey(5) & 0xFF
	if k ==27:
		break

cap.release()
cv2.destroyAllWindows()
