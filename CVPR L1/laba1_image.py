import numpy as np
import cv2

def new_image(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray= cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR)
    cv2.rectangle(gray, (50,50), (150,150),(200,200,5),2)
    cv2.line(gray, (100,200),(500,450),(230,0,15),2)
    return gray

cap = cv2.VideoCapture(0)

#while(True):
ret, frame = cap.read()
if ret:
    cv2.imshow('Video', frame)
    cv2.waitKey(0)
    cv2.imwrite("image.jpg", frame)

image = cv2.imread("image.jpg")
cv2.imshow("rectangle", new_image(image))
cv2.imwrite("new_image.jpg", new_image(image))
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()