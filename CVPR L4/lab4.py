import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from joblib import dump, load
import time

referenceType = 'battery'
desk = 'brisk'
init_desk = cv2.BRISK_create()
movie_name = 'videoBattery.mp4'
reference = cv2.imread('originals/referenceBattery.jpg', 0)
replace = cv2.imread('casperFace.jpg')

MIN_MATCH_COUNT = 12

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

descriptor_list = ['brisk']
kmeans = dict()
for i in descriptor_list:
    kmeans[i] = load(f'written_models/{i}_kmeans.joblib');
forest = dict()
for i in descriptor_list:
    forest[i] = load(f'written_models/{i}_forest.joblib')


brisk = cv2.BRISK_create()

descriptor = dict()
descriptor['brisk'] = brisk

replace2 = replace
replace = cv2.cvtColor(replace, cv2.COLOR_BGR2GRAY)


def create_bag(labels_list):
    res = np.zeros((10 * 4,))
    for i in labels_list:
        res[i] += 1
    return res


def preprocess_image(img, estimator):
    img = cv2.resize(img, (img.shape[0] // 4, img.shape[1] // 4))
    kp, des = descriptor[estimator].detectAndCompute(img, None)
    if (des is None):
        return None
    des = des.astype('float32')
    if (des.shape[0] > 500):
        des = des[:500, :]
    kmeans_labels = kmeans[estimator].predict(des)
    bag_of_words = create_bag(kmeans_labels)
    return bag_of_words


cap = cv2.VideoCapture(movie_name)

fps = []
count = 0
img = replace
img1 = reference
kp1, des1 = init_desk.detectAndCompute(img1, None)
start = time.time()
i = 0

while (cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:
        frame2 = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        vect = dict()

        vect[desk] = preprocess_image(frame, desk)

        temp = vect[desk].reshape((1, vect[desk].shape[0]))
        pred = logreg[desk].predict(temp)[0]


        if (pred == referenceType):

            kp2, des2 = init_desk.detectAndCompute(frame, None)
            matches = flann.knnMatch(des1, des2, k=2)

            good = []

            for i in matches:
                if (len(i) > 1):
                    m = i[0]
                    n = i[1]
                    if m.distance < 0.7 * n.distance:
                        good.append(m)

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                h, w = img.shape  # строим переобзование точек со 100 грн на 10 грн
                pts2 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                matrix = cv2.getPerspectiveTransform(pts2, dst)
                rows, cols = frame.shape
                result = cv2.warpPerspective(replace2, matrix, (cols, rows))
                result = cv2.addWeighted(result, 0.8, frame2, 0.5, 1)  # склеиваем картинки

                cv2.imshow('Frame', result)
                cv2.imwrite(f'resultIm{count}.jpg', result)

            else:
                print("Not enough matches are found :cry: - %d/%d" % (len(good), MIN_MATCH_COUNT))
                matchesMask = None
                cv2.imshow('Frame', frame2)
        else:
            cv2.imshow('Frame', frame2)
    else:
        break
    count += 1  # для вывода фпс
    if (time.time() - start > 1):
        fps.append(count)
        count = 0
        start = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
