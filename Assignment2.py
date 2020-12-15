
import os
from skimage import data, color, exposure
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
#-----------------------------------------INITIAL PARAMETERS-----------------------------


plt.figure(figsize=(16, 4))
plt.figure(figsize=(16, 4))
img_width = 64
img_height = 128
leftop = [16, 16]
rightbottom = [16 + img_width, 16 + img_height]


#--------------------DIRECTORY CONTAINING DATA FILES---------------------------------------


pos_img_dir = 'INRIA_Dataset_Samples/Train/pos/'
neg_img_dir = 'INRIA_Dataset_Samples/Train/neg/'
pos_img_files = os.listdir(pos_img_dir)
neg_img_files = os.listdir(neg_img_dir)
X = []
y = []

# ------------------------------LOADING DATASETS ------------------------------------

print('Loading  ' + str(len(pos_img_files)) + ' positive files')
print('Loading ' + str(len(neg_img_files)) + ' negative files')

# ------------------------computing HOG for entire training dataset---------------------------------

print("COMPUTING HOG for Positive Files of Training  Dataset")
for pos_img_file in pos_img_files:
    pos_filepath = pos_img_dir + pos_img_file
    pos_img = data.imread(pos_filepath, as_gray=True)
    pos_roi = pos_img[leftop[1]:rightbottom[1], leftop[0]:rightbottom[0]]
    fd = hog(pos_roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    X.append(fd)
    y.append(1)

print("COMPUTING HOG for Negative Files of Training  Dataset")
# print('start loading ' + str(len(neg_img_files)) + ' negative files')
for neg_img_file in neg_img_files:
    neg_filepath = neg_img_dir + neg_img_file
    neg_img = data.imread(neg_filepath, as_gray=True)
    neg_roi = neg_img[leftop[1]:rightbottom[1], leftop[0]:rightbottom[0]]
    fd = hog(neg_roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    X.append(fd)
    y.append(0)

print("HOG computation completed!")
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)


# ------------------------START LEARNING SVM---------------------------------


print('Start Learning SVM.')
lin_clf = svm.LinearSVC()
lin_clf.fit(X, y)
y_pred=lin_clf.predict(X)
# After training, check the accuracy using actual and predicted values.
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y, y_pred))
print('finish learning SVM.')
print('Fitness',lin_clf.fit(X,y))
print('Score',lin_clf.score(X,y))

disp = confusion_matrix(y, y_pred, labels=None)
print('Confusion Matrix showing TP,TN,FP,FN of SVM',disp)
target_names = ['TP', 'FN']
target_namesx = ['FN', 'TN']
sns.heatmap(confusion_matrix(y,lin_clf.predict(X)), annot=True,xticklabels=target_namesx,yticklabels=target_names)
plt.show()
joblib.dump(lin_clf, 'person_detector.pkl', compress=9)


# ------------------------START LEARNING Random Forest Classifier---------------------------------

print('Training Random Forest')
clf=RandomForestClassifier(random_state=0)
clf.fit(X,y)
y_pred=clf.predict(X)
print("Training Complete")
print("Accuracy:",metrics.accuracy_score(y, y_pred))
print('Finish learning Random Forest.')
print('Random Forest Fitness',clf.fit(X,y))
print('Random Forest Score',clf.score(X,y))
disp = confusion_matrix(y, y_pred, labels=None)
print('Confusion Matrix showing TP,TN,FP,FN of RF',disp)
a= f1_score(y, y_pred, average=None)
print('F1-Score',a)
sns.heatmap(confusion_matrix(y,clf.predict(X)), annot=True,xticklabels=target_namesx,yticklabels=target_names)
joblib.dump(clf, 'person_detector.pkl', compress=9)



# ------------------------TESTING & VISUALIZING RESULTS---------------------------------


pos_filepath = pos_img_dir + pos_img_files[2]
pos_img = data.imread(pos_filepath,as_grey=True)
neg_filepath = neg_img_dir + neg_img_files[2]
neg_img = data.imread(neg_filepath,as_grey=True)
pos_roi = pos_img[leftop[1]:rightbottom[1],leftop[0]:rightbottom[0]]
fd, pos_hog_image = hog(pos_roi, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=True)
neg_roi = neg_img[leftop[1]:rightbottom[1],leftop[0]:rightbottom[0]]
fd, neg_hog_image = hog(neg_roi, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=True)
pos_hog_image = exposure.rescale_intensity(pos_hog_image, in_range=(0, 0.1))
neg_hog_image = exposure.rescale_intensity(neg_hog_image, in_range=(0, 0.1))


plt.subplot(141).set_axis_off()
plt.imshow(pos_roi, cmap=plt.cm.gray)
plt.title('Positive image 0')
plt.subplot(142).set_axis_off()
plt.imshow(pos_hog_image, cmap=plt.cm.gray)
plt.title('Postive HOG')

plt.subplot(143).set_axis_off()
plt.imshow(neg_roi, cmap=plt.cm.gray)
plt.title('Negative image 0')
plt.subplot(144).set_axis_off()
plt.imshow(neg_hog_image, cmap=plt.cm.gray)
plt.title('Negative HOG')
plt.show()
