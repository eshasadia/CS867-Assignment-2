import cv2
import matplotlib.pyplot as plt
#matplotlib inline
#read images

img2=cv2.imread('roma_1.jpg',1)
img3=cv2.imread('roma_2.jpg',1)


#SIFT
sift=cv2.xfeatures2d.SIFT_create()

keypoints_2,descriptors_2=sift.detectAndCompute(img2,None)
keypoints_3,descriptors_3=sift.detectAndCompute(img3,None)
len(keypoints_2),len(keypoints_3)

bf=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
matches=bf.match(descriptors_2,descriptors_3)
matches=sorted(matches,key=lambda x:x.distance)
img4=cv2.drawMatches(img2,keypoints_2,img3,keypoints_3,matches[:10],img3,flags=2)
im4_resized = cv2.resize(img4, (1080, 640))
plt.title("SIFT MATCHING")
plt.imshow(img4)
plt.show()
