import cv2
from matplotlib import  pyplot as plt
from pylab import rcParams

img_1 = cv2.cvtColor(cv2.imread('building_1.jpg',1), cv2.COLOR_BGR2RGB)
img_2 = cv2.cvtColor(cv2.imread('building_3.jpg',1), cv2.COLOR_BGR2RGB)

surf=cv2.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SURF
kp_surf_1, des_surf_1 = surf.detectAndCompute(img_1, None)
kp_surf_2, des_surf_2 = surf.detectAndCompute(img_2, None)


rcParams['figure.figsize'] = 25, 25

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des_surf_1, des_surf_2)
matches = sorted(matches, key = lambda x:x.distance)
print(len(matches))
#lets show top 10 features matched
image_with_feature_matched = cv2.drawMatches(img_1, kp_surf_1, img_2, kp_surf_2, matches[:10], None, flags=2)
plt.imshow(image_with_feature_matched)
plt.title("SURF MATCHING")
plt.show()
