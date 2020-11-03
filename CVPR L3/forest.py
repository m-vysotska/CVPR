import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split

data = pd.read_excel("tablesift.xlsx")
data

X = data[['good matches', 'average distance']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

times_process=[]
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=1)

clf.fit(X_train,y_train)
start_time=time.time()
pred = clf.predict(X_test)
processing_time = time.time() - start_time
times_process.append(processing_time)

test_score = sum(pred == y_test)/len(pred)
print("sift, battery", test_score)
firstmistake = sum((pred == 0) & (y_test == 1))
print("first mistake", firstmistake)
secondmistake = sum((pred == 1) & (y_test == 0))
print("second mistake ", secondmistake)
right=sum((pred == 0) & (y_test == 0))+sum((pred == 1) & (y_test == 1))
print("right ",right)
avg_proc_time = np.mean(times_process)
print("Avg processing time is ", avg_proc_time, " seconds")


print("")
data = pd.read_excel("tablebrisk.xlsx")

X = data[['good matches', 'average distance']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

times_process=[]
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=1)

clf.fit(X_train,y_train)
start_time=time.time()
pred = clf.predict(X_test)
processing_time = time.time() - start_time
times_process.append(processing_time)
test_score = sum(pred == y_test)/len(pred)
print("brisk, battery" ,test_score)
firstmistake = sum((pred == 0) & (y_test == 1))
print("first mistake", firstmistake)
secondmistake = sum((pred == 1) & (y_test == 0))
print("second mistake ", secondmistake)
right=sum((pred == 0) & (y_test == 0))+sum((pred == 1) & (y_test == 1))
print("right ",right)
avg_proc_time = np.mean(times_process)
print("Avg processing time is ", avg_proc_time, " seconds")


