
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 5, 5

img = cv2.imread('book.jpg',0)
img1 = cv2.imread('book_person_holding.jpg',0)
# Initiate FAST detector
star = cv2.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(img,None)
kp1 = star.detect(img1,None)
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)
kp1, des1 = brief.compute(img1, kp1)

len(kp),len(kp1)
bf=cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
matches=bf.match(des,des1)
matches=sorted(matches,key=lambda x:x.distance)
img4=cv2.drawMatches(img,kp,img1,kp1,matches[:10],img1,flags=0)
plt.title('BRIEF MATCHING')
plt.imshow(img4)
plt.show()
