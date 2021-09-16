import cv2 as cv
import numpy as np 
########################################################################
#process_image
img = cv.imread("omr.png")
imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray,(5,5),10)
imgCanny = cv.Canny(imgGray,100,200)
########################################################################
#find_contours
countours,__ = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img,countours,-1,(0,255,0),1)
########################################################################
#rectangle_conutours
def rectangle_countours(countours):
    rect_cnts = []
    for c in countours:
        area = cv.contourArea(c)
        # print(area)
        if area > 80:
            peri = cv.arcLength(c,True)
            approx = cv.approxPolyDP(c,0.02*peri,True)
            # print(len(approx))
            if len(approx) == 4:
                rect_cnts.append(c)   
    rect_cnts = sorted(rect_cnts,key = cv.contourArea,reverse=True)
    return rect_cnts
########################################################################
def get_conrner_point(c):
    peri = cv.arcLength(c,True)
    approx = cv.approxPolyDP(c,0.02*peri,True)
    return approx
########################################################################    
def four_points(points):
    points = points.reshape(4,2)
    new_points = np.zeros((4,1,2),dtype="float32")
    s = points.sum(axis = 1)
    print(points)
    print(s)
    
    new_points[0] = points[np.argmin(s)]
    new_points[3] = points[np.argmax(s)]
    diff = np.diff(points, axis = 1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    print(diff)
    return new_points
########################################################################
rect_cnts = rectangle_countours(countours)
biggest_countours = get_conrner_point(rect_cnts[0])
if biggest_countours.size != 0:
    # cv.drawContours(img,biggest_countours,-1,(255,0,0),20)
    # four_points(biggest_countours)    
    biggest_countours = four_points(biggest_countours)
# # 559 x 842 
    point_1 = np.float32(biggest_countours)
    point_2 = np.float32([[0,0],[300,0],[0,400],[300,400]])
    matrix = cv.getPerspectiveTransform(point_1,point_2)
    img_warp = cv.warpPerspective(img,matrix,(300,400))

imgArray = ([img,imgGray,imgBlur,imgCanny])
img_warp_Gray = cv.cvtColor(img_warp,cv.COLOR_BGR2GRAY)
img_warp_Blur = cv.GaussianBlur(img_warp,(5,5),10)
img_warp_Canny = cv.Canny(img_warp,100,200)

cv.imshow('image',img_warp)
cv.waitKey(0)
print (hahaha)
