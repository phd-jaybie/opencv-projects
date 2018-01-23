# Name: Always on 2D object detection 
# Author: Jaybie A. de Guzman, 		Date: 22-Jan-2018
# Description:
# Adapted from the image detection sample code from the opencv-python
# tutorial at http://docs.opencv.org/3.3.0/d1/de0/tutorial_py_feature_homography.html

# Version info:	4.0 - this version uses ORB for detection and hopefully for SLAM.

#			2.0 - this version is the stable implementation with argument handling.

#			1.1 - 	this version shows information of the distance calculation
#			of the query and reference descriptors which was used for matching.

import sys
import numpy as np
import cv2
import time
#from matplotlib import pyplot as plt

if __name__ == '__main__':
	# some global variables
	search_params = dict(checks = 20) # this is for the flann-based matcher
	MIN_MATCH_COUNT = 10 # very relaxed matching at 10 matches minimum

	if len(sys.argv) > 1:

		# from arguments, check what feature detector algorithm
		if 'sift' in sys.argv:
			detector = cv2.xfeatures2d.SIFT_create()
			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
			print("Using SIFT algorithm")
		elif 'orb' in sys.argv:
			detector = cv2.ORB_create()
			print("Using ORB for detection.")
			descriptor = cv2.xfeatures2d.SIFT_create()
			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
			print("Using SIFT for decription.")
		else:
			detector = cv2.BRISK_create()
			FLANN_INDEX_LSH = 6
			index_params = dict(algorithm = FLANN_INDEX_LSH,\
			table_number = 6, key_size = 12, multi_proble_level = 1)
			print("Using BRISK algorithm")

		# from arguments, check what matcher is used
		if 'flann' in sys.argv:
			matcher = cv2.FlannBasedMatcher(index_params, search_params)
			print("Using flann-based matcher")
		else:
			matcher = cv2.BFMatcher() #for regular descriptors
			#matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # for ORB keypoints
			print("Using brute force matcher")
		
		# from arguments, check matching algorithm
		if 'match' in sys.argv:
			print("Using regular matching")
		else:
			print("Using knn matching")
	else:
		detector = cv2.xfeatures2d.SIFT_create()
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
		print("Default: Using SIFT algorithm")
		matcher = cv2.FlannBasedMatcher(index_params, search_params)
		print("Default: Using flann-based matcher")
		print("Default: Using knn matching")

	if 'jpg' in sys.argv[-1] or 'png' in sys.argv[-1]:
		img1 = cv2.imread(str(sys.argv[-1]),0)          # queryImage
	else:
		# change this if needs to query multiple images
		img1 = cv2.imread('train.jpg',0)          # queryImage

	cap = cv2.VideoCapture(0)

	frameCount = 0
	
	# find the keypoints and descriptors from the training image with SIFT
	kp1, des1 = detector.detectAndCompute(img1,None)
	#kp1 = detector.detect(img1,None)
	#__, des1 = descriptor.compute(img1, kp1)

	#print("Frame, t_detect, t_match, t_sort , t_transform, Training Descriptors, Query Descriptors, Matches, Min Distance, Max Distance")

	while(frameCount<100):
		# get each frame
		_, raw = cap.read()
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), not necessary
		# ox,oy = frame.shape
		frame = cv2.resize(raw,(0,0),fx=0.5,fy=0.5)
		fh,fw = frame.shape[:2]

		t_start = time.clock()
		frameCount = frameCount + 1

		# find features on the frame with SIFT/ORB
		kp2, des2 = detector.detectAndCompute(frame,None)
		#kp2 = detector.detect(frame,None)
		#__, des2 = descriptor.compute(frame, kp2)
		t_detect = time.process_time()

		if 'match' in sys.argv:
			good = matcher.match(des1,des2)	
			t_match = time.process_time()
		else:
			#change this if want to use any k
			matches = matcher.knnMatch(des1,des2,k=2)
			t_match = time.process_time()
			 
			# store all the good matches as per Lowe's ratio test.
			good = []
			distances = []
			
			#for m,n in matches:
			#	distances.append(m.distance)
			#	if m.distance < 0.75*n.distance:
			#		good.append(m)

		good = sorted(good, key = lambda x:x.distance)
		t_sort = time.process_time()

		mask2 = np.zeros(frame.shape, np.uint8)

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
			flood_mask = np.zeros((fh+2,fw+2), np.uint8)
			cv2.drawContours(frame,[np.int32(dst)],-1,(255,0,0),3)
			cv2.drawContours(mask2,[np.int32(dst)],-1,(255,0,0),3)
			cv2.floodFill(mask2,flood_mask,(0,0),(255,255,255))
			cv2.drawKeypoints(frame,kp2,frame,color=(0,255,0), flags=4)
			res = frame | mask2
		else:
			res = mask2
			matchesMask = None
		
		t_transform = time.process_time()
		#print("%d, %.5f, %.5f, %.5f, %.5f, %d, %d, %d, %.5f, %.5f" % (frameCount, \
			#t_detect-t_start, t_match-t_detect, t_sort-t_match, t_transform-t_sort, \
			#len(des1),len(des2),len(good),good[0].distance,good[-1].distance))

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
        
		frameRate = 1/(t_transform - t_start)
		frameRateText = "Frame Rate: " + str(frameRate)
		cv2.putText(res, frameRateText, (10,30), cv2.FONT_HERSHEY_SIMPLEX, \
			0.5, (10,10,10),1, cv2.LINE_AA)

		cv2.imshow('Detected -> Sanitized', res)
		cv2.imshow('Matches',img3) # shows the matching lines for checking

#		plt.hist(distances,normed=False, bins = 30)
#		plt.ylabel('Probability')

		k = cv2.waitKey(5) & 0xFF
		if k ==27:
			cv2.imwrite("detected.png", res)
			cv2.imwrite("matches.png", img3)
			cv2.imwrite("raw.png", raw)
			break

	cap.release()
	cv2.destroyAllWindows()
