import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import pandas as pd

good = []
good_matches = []
time_process = []
i_arr = []
avg_distance = []


def sift_match(img1_path, img2_path):
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)  # trainImage
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
        if m.distance < 0.75 * n.distance:
            good.append([m])
            good_metric += 1
            metric2.append((m.distance + n.distance) / 2)

    if (len(matches) == 0):
        good_matches[i] = 0
    else:
        metric1 = good_metric / len(matches)
    good_matches.append(metric1)

    # metric2
    if (len(metric2) != 0):
        metric2 = np.array(metric2).mean()
    else:
        metric2 = 0
    avg_distance.append(metric2)

    # cv.drawMatchesKnn expects list of lists as matches.
    # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()
    print(i)


for i in range(1, 121):
    i_arr.append(i)
    sift_match("reference.jpg", f"img{i}.jpg")

df = pd.DataFrame({'img': i_arr, 'time': time_process, 'good matches': good_matches, 'average distance': avg_distance})
df.to_excel('./tablebriskdarth.xlsx')