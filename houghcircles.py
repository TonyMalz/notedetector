import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('entchen.png', cv2.IMREAD_GRAYSCALE)
imgblur = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
fig = plt.figure()
circles = cv2.HoughCircles(imgblur, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=10, minRadius=0, maxRadius=5)
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),1,(0,0,255),3)

cv2.imshow('detected circles', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
