
import numpy as np
import cv2
from matplotlib import pyplot as plt

# you can use following code to increase or decrease your figure size
from pylab import rcParams
rcParams['figure.figsize'] = 5, 5

img = cv2.imread('roma_1.jpg',0)
img1=cv2.imread('roma_2.jpg',0)
print(img.shape)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp_orb, des_orb = orb.detectAndCompute(img, None)

kp_orb1, des_orb1 = orb.detectAndCompute(img1, None)

len(kp_orb),len(kp_orb1)

# Match descriptors.
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_orb,des_orb1)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img,kp_orb,img1,kp_orb1,matches[:10],img1, flags=2,matchColor=(0,0,255))

plt.imshow(img3)
plt.title("ORB MATCHING")
plt.show()
