# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# img = cv2.imread('building_1.jpg',0)
# img1 = cv2.imread('building_2.jpg',0)
# # Initiate FAST object with default values
# fast = cv2.FastFeatureDetector_create()
# # find and draw the keypoints
# kp1= fast.detect(img,None)
# kp2= fast.detect(img1,None)
# img2 = cv2.drawKeypoints(img, kp1, None, color=(255,0,0))
# img3 = cv2.drawKeypoints(img, kp2, None, color=(255,0,0))
# # Print all default params
# print( "Threshold: {}".format(fast.getThreshold()) )
# print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
# print( "neighborhood: {}".format(fast.getType()) )
# print( "Total Keypoints for img1 with nonmaxSuppression: {}".format(len(kp1)) )
# print( "Total Keypoints for img2 with nonmaxSuppression: {}".format(len(kp2)) )
# #cv2.imwrite('fast_true.png',img2)
# plt.imshow(img2)
# plt.show()
# # Disable nonmaxSuppression
# fast.setNonmaxSuppression(0)
# kp = fast.detect(img,None)
# print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
# img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
# # cv2.imwrite('fast_false.png',img3)
#
#
# # keypoints_1,descriptors_1=sift.detectAndCompute(img1,None)
# # keypoints_2,descriptors_2=sift.detectAndCompute(img2,None)
# # keypoints_3,descriptors_3=sift.detectAndCompute(img3,None)
# # len(keypoints_1),len(keypoints_2),len(keypoints_3)
# bf=cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
# matches=bf.match(kp1,descriptors_3)
# matches=sorted(matches,key=lambda x:x.distance)
# img4=cv2.drawMatches(img2,keypoints_2,img3,keypoints_3,matches[:50],img3,flags=2)
# plt.imshow(img3)
# plt.show()


import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('roma_1.jpg',1)
img1 = cv2.imread('roma_2.jpg',1)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img,None)
kp1= fast.detect(img1,None)
#img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

bf=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
matches=bf.match(kp,kp1)
matches=sorted(matches,key=lambda x:x.distance)
img4=cv2.drawMatches(img,kp,img1,kp1,matches[:10],img1,flags=2,matchColor=(0,0,255))
plt.title("FAST MATCHING")
plt.imshow(img4)
plt.show()

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

# # Disable nonmaxSuppression
# fast.setNonmaxSuppression(0)
# kp = fast.detect(img,None)
# print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
# img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
# ax1.axis('off')
# ax1.imshow(img2, cmap=plt.cm.gray)
# ax1.set_title('With NMS')
# ax2.axis('off')
# ax2.imshow(img3)
# ax2.set_title('Without NMS')
# plt.show()

# Non-maximum suppression:- is a class of algorithms to select one entity out of many overlapping entities.