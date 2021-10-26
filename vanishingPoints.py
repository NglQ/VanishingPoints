# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import math

img = cv2.imread('biblioteca.jpg')
imgOrig = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imwrite('gray.jpg',gray)

gray = cv2.GaussianBlur(gray,(3,3),0,0)
# gray = cv2.medianBlur(gray,5)

filter45=np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]);

grayFilt = cv2.filter2D(gray,-1,filter45)

cv2.imwrite('grayFilt.jpg',grayFilt)

edges = cv2.Canny(grayFilt,10,400,apertureSize = 3)

cv2.imwrite('canny.jpg',edges)

#TODO: find an elegant way to estimate the threshold since there is not a way to have the peaks
#otherwise implement Hough transform from scratch (hopefully not!!!)

data = []
lines = cv2.HoughLinesP(edges,rho = 1,theta = np.pi/180,threshold = 50,minLineLength=14,maxLineGap=300)
if lines is not None and len(lines) > 0:
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            #cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            distCat =cv2.norm(np.array([x1,y1]),np.array([x1,y2]),cv2.NORM_L2)
            distHyp = cv2.norm(np.array([x1,y1]),np.array([x2,y2]),cv2.NORM_L2)
            angle = np.arcsin(distCat/distHyp)
            data.append(np.array([angle,np.cos(2*angle), np.sin(2*angle)]))

    #cv2.imwrite('houghlines3.jpg',img)

#TODO: filter away every almost parallel lines (use k means with adaptive k)
#implement xmeans instead of using kmeans

    data = np.array(data, dtype=np.float32)

    #print(len(data)
    #print(data)
    nClusters = 10

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(data, nClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#TODO: find representative lines for each cluster

    repLines = []

    for i in range(nClusters):
        clusterIdx = (label == i)
        clusterData = [data[x] for x in range(data.shape[0]) if clusterIdx[x]]
        clusterLines = lines[clusterIdx]
        clDiff = [cv2.norm(np.array(x - center[i]),cv2.NORM_L2) for x in clusterData]
        minLine = clDiff.index(min(clDiff))
        
        cv2.line(img,(clusterLines[minLine][0],clusterLines[minLine][1]),(clusterLines[minLine][2],clusterLines[minLine][3]),(0,255,0),2)
        
        #TODO: find a better way to filter 90° angles
        if center[i][1] < 0.984807753 or center[i][1] > -0.984807753: 
            repLines.append(clusterLines[minLine])
        
        imgTemp = imgOrig.copy()
        for cll in clusterLines:
            #print(cll)
            cv2.line(imgTemp,(cll[0],cll[1]),(cll[2],cll[3]),(0,0,255),2)
        cv2.imwrite('houghlinesCls'+  str(i) +'.jpg',imgTemp)
            
            #print(clDiff)
            
            #print(minLine)
            

    cv2.imwrite('houghlinesRep.jpg',img)
#TODO: find intersection among the lines generated from the real lines closest to the centroids
    print(len(repLines))
    intersections = []
    for li1 in range(len(repLines)):
        ln1 = repLines[li1]
        m1 = (ln1[3] - ln1[1]) / (ln1[2] - ln1[0]) 
        q1 = ln1[1] - m1 * ln1[0]
        for li2 in range(li1,len(repLines)):
            ln2 = repLines[li2]
            m2 = (ln2[3] - ln2[1]) / (ln2[2] - ln2[0])
            q2 = ln2[1] - m2 * ln2[0]
            
            xint = (q1-q2) / (m2-m1)
            yint = m1 * xint + q1
            #print(xint,yint)
            if not math.isnan(xint) and not math.isnan(yint):
                img = cv2.circle(img, (round(xint),round(yint)), radius = 2, color = (255,0,0), thickness = -1)
                intersections.append(np.array([xint, yint]))
    
    #print(intersections)
    cv2.imwrite("intersections.jpg", img)
    
# print(center)
# print(label)
# print(ret)

#TODO: filter away non-reliable results (good luck with that!!!)
    
    
    
    
    
    
else:
    print("No lines found!!!")