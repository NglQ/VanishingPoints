# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import math

img = cv2.imread('corridoio2.jpg')
imgOrig = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imwrite('gray.jpg',gray)

gray = cv2.GaussianBlur(gray,(3,3),0,0)

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
            data.append(np.array([np.cos(angle), np.sin(angle)]))

    #cv2.imwrite('houghlines3.jpg',img)

#TODO: filter away every almost parallel lines (use k means with adaptive k)
#implement xmeans instead of using kmeans

    data = np.array(data, dtype=np.float32)

    nClusters = 10

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(data, nClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    
#TODO: find representative lines for each cluster

    repLines = []
    intersections = []
    
    for i in range(nClusters):
        clusterIdx = (label == i)
        clusterData = [data[x] for x in range(data.shape[0]) if clusterIdx[x]]
        clusterLines = lines[clusterIdx]
        #clDiff = [cv2.norm(np.array(x - center[i]),cv2.NORM_L2) for x in clusterData]
        #minLine = clDiff.index(min(clDiff))
        
        #cv2.line(img,(clusterLines[minLine][0],clusterLines[minLine][1]),(clusterLines[minLine][2],clusterLines[minLine][3]),(0,255,0),2)
        
        #TODO: find a better way to filter 90° angles
        if center[i][1] < 0.984807753 or center[i][1] > -0.984807753: 
            for li1 in range(len(clusterLines)):
                ln1 = clusterLines[li1]
                m1 = (ln1[3] - ln1[1]) / (ln1[2] - ln1[0]) 
                q1 = ln1[1] - m1 * ln1[0]
                for li2 in range(li1,len(clusterLines)):
                    ln2 = clusterLines[li2]
                    m2 = (ln2[3] - ln2[1]) / (ln2[2] - ln2[0])
                    q2 = ln2[1] - m2 * ln2[0]
            
                    if m2-m1 != 0.0: 
                        xint = (q1-q2) / (m2-m1)
                        yint = m1 * xint + q1
                    else:
                        xint = np.inf
                        yint = np.inf
            #print(xint,yint)
                    if not math.isnan(xint) and not math.isnan(yint) and not math.isinf(xint) and not math.isinf(yint):
                        img = cv2.circle(img, (round(xint),round(yint)), radius = 1, color = (255,0,0), thickness = -1)
                        intersections.append(np.array([xint, yint]))
            #repLines.append(clusterLines[minLine])
        
        imgTemp = imgOrig.copy()
        for cll in clusterLines:
            #print(cll)
            cv2.line(imgTemp,(cll[0],cll[1]),(cll[2],cll[3]),(0,0,255),2)
        cv2.imwrite('houghlinesCls'+  str(i) +'.jpg',imgTemp)
            
    cv2.imwrite("intersections.jpg", img)
    
#TODO: filter away non-reliable results (good luck with that!!!)
    windows = []
    imgShape = gray.shape 
    winSize = int(min(imgShape) / 10)
    #print(winSize)
    for intersection in intersections:
        # xmin = intersection[0] - winSize if intersection[0] - winSize > 0.0  else 0.0
        # ymin = intersection[1] - winSize if intersection[1] - winSize > 0.0 else 0.0
        # xmax = intersection[0] + winSize if intersection[0] + winSize < imgShape[0] else imgShape[0]
        # ymax = intersection[1] + winSize if intersection[1] + winSize < imgShape[1] else imgShape[1]
        xmin = intersection[0] - winSize #if intersection[0] - winSize > 0.0  else 0.0
        ymin = intersection[1] - winSize #if intersection[1] - winSize > 0.0 else 0.0
        xmax = intersection[0] + winSize #if intersection[0] + winSize < imgShape[0] else imgShape[0]
        ymax = intersection[1] + winSize #if intersection[1] + winSize < imgShape[1] else imgShape[1]
        windows.append([xmin, ymin, xmax, ymax])
        #windows.append([xmax, ymax, xmin, ymin])
    
    intersectionVoting = np.zeros(len(intersections))
    #WARNING: there is somethin wrong with this part of code:
        #Apparently the coordinates of the intersections are not meant to be used as a index for a matrix (image)
    histImage = np.zeros(gray.shape)
    for intersection in intersections:
        xIntHist = int(intersection[0]) 
        yIntHist = int(intersection[1])
        
        #TODO: find a good way to set points where they should be
        if xIntHist < histImage.shape[1] and xIntHist > 0 and yIntHist < histImage.shape[0] and yIntHist > 0:
        # xIntHist = xIntHist if xIntHist < img.shape[0] else img.shape[0] - 1
        # yIntHist = yIntHist if yIntHist < img.shape[1] else img.shape[1] - 1
        # xIntHist = xIntHist if xIntHist > 0 else 0
        # yIntHist = yIntHist if yIntHist > 0 else 0
        
            #print(xIntHist, yIntHist)
        
            #histImage[xIntHist, yIntHist] =+255
            histImage[yIntHist,xIntHist] =+255
    
    for wndIdx in range(len(windows)):
        xMin1 = int(np.floor(windows[wndIdx][0]))
        xMax1 = int(np.floor(windows[wndIdx][2]))
        yMin1 = int(np.floor(windows[wndIdx][1]))
        yMax1 = int(np.floor(windows[wndIdx][3]))
        
        #print(xMin1,xMax1,yMin1,yMin1)
        
        #if abs(xMax1 - xMin1) != 0 and abs(yMax1 - yMin1) != 0:   
            #print(wndIdx)
            #print(range(xMin1,xMax1))
            #print(range(yMin1,yMax1))
        #print(histImage[xMin1:xMax1,yMin1:yMax1])
        if len(histImage[xMin1:xMax1,yMin1:yMax1]) != 0:
            intersectionVoting[wndIdx] = sum(sum(histImage[int(np.floor(windows[wndIdx][0])):int(np.floor(windows[wndIdx][2])),int(np.floor(windows[wndIdx][1])):int(np.floor(windows[wndIdx][3]))]))
        else:
            intersectionVoting[wndIdx] = 0
    cv2.imwrite("histImage.jpg", histImage)
    cv2.imwrite("histImageSum.jpg", histImage + gray)
    
    
    # for interIdx in range(len(intersections)):
    #     for windIdx in range(len(windows)):
    #         if intersections[interIdx][0] < windows[windIdx][2] and intersections[interIdx][0] > windows[windIdx][0] and intersections[interIdx][1] < windows[windIdx][2] and intersections[interIdx][1] > windows[windIdx][1]:
    #             intersectionVoting[windIdx] += 1.0
    
    #print(intersectionVoting.argmax(max(intersectionVoting)))
    #print(np.argmax(intersectionVoting))
    amIntVot = np.argmax(intersectionVoting)
    
    print(windows[amIntVot])
    win = windows[amIntVot]
    
    cpwind = imgOrig.copy()
    
    cpwind[int(win[0]):int(win[2]),int(win[1]):int(win[3]),0] = 255
    cpwind[int(win[0]):int(win[2]),int(win[1]):int(win[3]),1] = 255
    cpwind[int(win[0]):int(win[2]),int(win[1]):int(win[3]),2] = 255
    cv2.imwrite("bestWin.jpg", cpwind)
    
    
else:
    print("No lines found!!!")
    
#TODO: use genetic algorithms to estimate the parameter involved in the vanishing points algorithm
#TODO: find a good fitness function
#TODO: Refactor required