import numpy as np
import pandas as pd
import time

data = pd.read_excel("tablebrisk.xlsx")
print("brisk")
zeroes = data[data['label'] == 0]
# for i in range(4):
#    data = data.append(zeroes)
data

X = data['good matches']
y = data['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


def my_model_predict(matches, a):
    return (matches > a).astype(int)


# training model
max_score = 0
par = 0
times_process = []
for a in np.arange(0, 0.05, 0.001):
    start_time = time.time()
    pred = my_model_predict(X_train, a)
    processing_time = time.time() - start_time
    times_process.append(processing_time)

    score = sum(pred == y_train) / len(pred)
    if score >= max_score:
        max_score = score
        par = a
print("max score", max_score, "number", par)
avg_proc_time = np.mean(times_process)
print("Avg processing time is ", avg_proc_time, " seconds")
a = par

# testing model:
pred = my_model_predict(X_test, a)
test_score = sum(pred == y_test) / len(pred)
print("test ", test_score)

firstmistake = sum((pred == 0) & (y_test == 1))
print("first mistake", firstmistake)
secondmistake = sum((pred == 1) & (y_test == 0))
print("second mistake ", secondmistake)
right = sum((pred == 0) & (y_test == 0)) + sum((pred == 1) & (y_test == 1))
print("right ", right)

print("")
data = pd.read_excel("tablesift.xlsx")
print("sift")
zeroes = data[data['label'] == 0]
# for i in range(4):
#    data = data.append(zeroes)
data

X = data['good matches']
y = data['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


def my_model_predict(matches, a):
    return (matches > a).astype(int)


# training model
max_score = 0
par = 0
times_process = []
for a in np.arange(0, 0.05, 0.001):
    start_time = time.time()
    pred = my_model_predict(X_train, a)
    processing_time = time.time() - start_time
    times_process.append(processing_time)

    score = sum(pred == y_train) / len(pred)
    if score >= max_score:
        max_score = score
        par = a
print("max score", max_score, "number", par)
avg_proc_time = np.mean(times_process)
print("Avg processing time is ", avg_proc_time, " seconds")
a = par

# testing model:
pred = my_model_predict(X_test, a)
test_score = sum(pred == y_test) / len(pred)
print("test ", test_score)

firstmistake = sum((pred == 0) & (y_test == 1))
print("first mistake", firstmistake)
secondmistake = sum((pred == 1) & (y_test == 0))
print("secondmistake ", secondmistake)
right = sum((pred == 0) & (y_test == 0)) + sum((pred == 1) & (y_test == 1))
print("right ", right)
