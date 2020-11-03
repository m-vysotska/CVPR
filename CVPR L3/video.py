import numpy as np
import pandas as pd

data = pd.read_excel("./tablebrisk.xlsx")

X = data['good matches']
y = data['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


def my_model_predict(matches, a):
    return (matches > a).astype(int)


# training model
max_score = 0
par = 0
for a in np.arange(0, 0.05, 0.001):
    pred = my_model_predict(X_train, a)

    score = sum(pred == y_train) / len(pred)
    if score >= max_score:
        max_score = score
        par = a
#print(max_score, par)
a = par

# testing model:
pred = my_model_predict(X_test, a)
test_score = sum(pred == y_test) / len(pred)
print("score with my_classifier ", test_score)


X = data[['good matches', 'average distance']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(max_depth=2, random_state=1)

clf_rf.fit(X_train, y_train)

pred = clf_rf.predict(X_test)
test_score = sum(pred == y_test) / len(pred)
print("Random forest ", test_score)


X = data[['good matches', 'average distance']]
y = data['label']


# In[ ]:


import cv2 as cv
import time

good = []
good_matches = []
time_process = []
i_arr = []
avg_distance = []


def brisk_match(img1_path, img2):
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)# queryImage  # trainImage
    # Initiate SIFT detector
    brisk = cv.BRISK_create()
    # find the keypoints and descriptors with SIFT
    start_time = time.time()
    kp1, des1 = brisk.detectAndCompute(img1, None)
    kp2, des2 = brisk.detectAndCompute(img2, None)
    processing_time = time.time() - start_time
    time_process.append(processing_time)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test

    good_metric = 0
    metric2 = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
            good_metric += 1
            metric2.append((m.distance + n.distance) / 2)

    if (len(matches) == 0):
        metric1 = 0
    else:
        metric1 = good_metric / len(matches)
    good_matches.append(metric1)

    # metric2
    if (len(metric2) != 0):
        metric2 = np.array(metric2).mean()
    else:
        metric2 = 0
    avg_distance = metric2

    # cv.drawMatchesKnn expects list of lists as matches.
    #img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #plt.imshow(img3), plt.show()
    kp2 = [i.pt for i in kp2]
    return metric1, metric2, np.mean(kp2, axis = 0)
    # print(i, good_metric, len(matches))


# for i in range (1,2):
# brisk_match ("photo_2020-10-04_13-32-55.jpg","photo_2020-10-04_13-32-57.jpg")
# i_arr.append(i)
def model_predict(img):
    a, b, kp_mean = brisk_match("reference.jpg", img)
    pred = clf_rf.predict( np.array([a, b]).reshape(1, 2)  )[0]
    return pred, a, b, kp_mean

#print(model_predict(cv.imread(f"img110.jpg", cv.IMREAD_GRAYSCALE))[0])
# cv.imread(f"img110.jpg", cv.IMREAD_GRAYSCALE)

import cv2
cap = cv2.VideoCapture('video.mp4')
count = 0
preds = []
times_process = []
while cap.isOpened():
    ret, frame1 = cap.read()
    if ret:
        frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        pred, a, b, kp_mean = model_predict(frame)
        processing_time = time.time() - start_time
        times_process.append(processing_time)
        if pred == 1:
            frame1 = cv2.circle(frame1, tuple(kp_mean.astype(int)),4, (100,255,255), 4)
        preds.append(pred)
        frame1 = cv2.resize(frame1, (400, 600))
        #cv2.imshow('frame1', frame1)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit q to exit
            break
    else:
        break
avg_proc_time = np.mean(times_process)
print("Avg processing time is ", avg_proc_time, " seconds")
cap.release()
print("frames with an item: ", sum(preds)/len(preds))

import numpy as np
import pandas as pd

data = pd.read_excel("./tablebrisk.xlsx")

X = data['good matches']
y = data['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


def my_model_predict(matches, a):
    return (matches > a).astype(int)


# training model
max_score = 0
par = 0
for a in np.arange(0, 0.05, 0.001):
    pred = my_model_predict(X_train, a)

    score = sum(pred == y_train) / len(pred)
    if score >= max_score:
        max_score = score
        par = a
#print(max_score, par)
a = par

# testing model:
pred = my_model_predict(X_test, a)
test_score = sum(pred == y_test) / len(pred)
print("score with my_classifier ", test_score)


X = data[['good matches', 'average distance']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(max_depth=2, random_state=1)

clf_rf.fit(X_train, y_train)

pred = clf_rf.predict(X_test)
test_score = sum(pred == y_test) / len(pred)
print("Random forest ", test_score)


X = data[['good matches', 'average distance']]
y = data['label']


# In[ ]:

print("")
print("sift")

good = []
good_matches = []
time_process = []
i_arr = []
avg_distance = []


def sift_match(img1_path, img2):
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)# queryImage  # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    start_time = time.time()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    processing_time = time.time() - start_time
    time_process.append(processing_time)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test

    good_metric = 0
    metric2 = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
            good_metric += 1
            metric2.append((m.distance + n.distance) / 2)

    if (len(matches) == 0):
        metric1 = 0
    else:
        metric1 = good_metric / len(matches)
    good_matches.append(metric1)

    # metric2
    if (len(metric2) != 0):
        metric2 = np.array(metric2).mean()
    else:
        metric2 = 0
    avg_distance = metric2

    # cv.drawMatchesKnn expects list of lists as matches.
    #img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #plt.imshow(img3), plt.show()
    kp2 = [i.pt for i in kp2]
    return metric1, metric2, np.mean(kp2, axis = 0)
    # print(i, good_metric, len(matches))


# for i in range (1,2):
# brisk_match ("photo_2020-10-04_13-32-55.jpg","photo_2020-10-04_13-32-57.jpg")
# i_arr.append(i)
def model_predict(img):
    a, b, kp_mean = sift_match("reference.jpg", img)
    pred = clf_rf.predict( np.array([a, b]).reshape(1, 2)  )[0]
    return pred, a, b, kp_mean

#print(model_predict(cv.imread(f"img110.jpg", cv.IMREAD_GRAYSCALE))[0])
# cv.imread(f"img110.jpg", cv.IMREAD_GRAYSCALE)

import cv2
cap = cv2.VideoCapture('video.mp4')
count = 0
preds = []
times_process = []
while cap.isOpened():
    ret, frame1 = cap.read()
    if ret:
        frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        pred, a, b, kp_mean = model_predict(frame)
        processing_time = time.time() - start_time
        times_process.append(processing_time)
        if pred == 1:
            frame1 = cv2.circle(frame1, tuple(kp_mean.astype(int)),4, (100,255,255), 4)
        preds.append(pred)
        frame1 = cv2.resize(frame1, (400, 600))
        #cv2.imshow('frame1', frame1)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit q to exit
            break
    else:
        break
avg_proc_time = np.mean(times_process)
print("Avg processing time is ", avg_proc_time, " seconds")
cap.release()
print("frames with an item: ", sum(preds)/len(preds))