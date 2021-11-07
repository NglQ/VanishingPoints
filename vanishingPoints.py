import cv2
import numpy as np
import math

#TODO:
    #1) find an elegant way to estimate the threshold since there is not a way to have the peaks
            #otherwise implement Hough transform from scratch (hopefully not!!!)
    #2) find a better way to filter 90° angles
    #3) filter away non-reliable results (good luck with that!!!)
    #4) use genetic algorithms to estimate the parameter involved in the vanishing points algorithm
        #4.1) find a good fitness function
    #5) Refactor required
    

def readImage(pathToImage):
    img = cv2.imread(pathToImage)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgOrig = img.copy()
    cv2.imwrite('gray.jpg',gray)
    return img, gray, imgOrig

def preprocessing(gray):
    gray = cv2.GaussianBlur(gray,(3,3),0,0)
    filter45=np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]);
    grayFilt = cv2.filter2D(gray,-1,filter45)
    cv2.imwrite('grayFilt.jpg',grayFilt)
    edges = cv2.Canny(grayFilt,10,400,apertureSize = 3)
    return edges

def detectLines(edges):
    data = []
    lines = cv2.HoughLinesP(edges,rho = 1,theta = np.pi/180,threshold = 50,minLineLength=14,maxLineGap=300)
    if lines is not None and len(lines) > 0:
        for i in range(len(lines)):
            for x1,y1,x2,y2 in lines[i]:
                distCat =cv2.norm(np.array([x1,y1]),np.array([x1,y2]),cv2.NORM_L2)
                distHyp = cv2.norm(np.array([x1,y1]),np.array([x2,y2]),cv2.NORM_L2)
                angle = np.arcsin(distCat/distHyp)
                data.append(np.array([np.cos(angle), np.sin(angle)]))

    data = np.array(data, dtype=np.float32)
    return lines, data

def clustering(data,nClusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(data, nClusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return ret, label, center

def detectIntersections(nClusters, label, center, data, lines, img, imgOrig):
    intersections = []
    
    for i in range(nClusters):
        clusterIdx = (label == i)
        clusterLines = lines[clusterIdx]
        
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
                    if not math.isnan(xint) and not math.isnan(yint) and not math.isinf(xint) and not math.isinf(yint):
                        img = cv2.circle(img, (round(xint),round(yint)), radius = 1, color = (255,0,0), thickness = -1)
                        intersections.append(np.array([xint, yint]))
        
        imgTemp = imgOrig.copy()
        for cll in clusterLines:
            cv2.line(imgTemp,(cll[0],cll[1]),(cll[2],cll[3]),(0,0,255),2)
        cv2.imwrite('houghlinesCls'+  str(i) +'.jpg',imgTemp)
            
    cv2.imwrite("intersections.jpg", img)
    return intersections

def detectWindows(gray,intersections):
    windows = []
    imgShape = gray.shape 
    winSize = int(min(imgShape) / 10)
    
    for intersection in intersections:
        xmin = intersection[0] - winSize
        ymin = intersection[1] - winSize
        xmax = intersection[0] + winSize
        ymax = intersection[1] + winSize
        windows.append([xmin, ymin, xmax, ymax])
    return windows

def getIntersectionDescriptors(intersections,gray,windows):
    
    intersectionVoting = np.zeros(len(intersections))
    histImage = np.zeros(gray.shape)
    for intersection in intersections:
        xIntHist = int(intersection[0]) 
        yIntHist = int(intersection[1])
        
        if xIntHist < histImage.shape[1] and xIntHist > 0 and yIntHist < histImage.shape[0] and yIntHist > 0:
            histImage[yIntHist,xIntHist] =+1
    
    for wndIdx in range(len(windows)):
        xMin1 = int(np.floor(windows[wndIdx][0]))
        xMax1 = int(np.floor(windows[wndIdx][2]))
        yMin1 = int(np.floor(windows[wndIdx][1]))
        yMax1 = int(np.floor(windows[wndIdx][3]))
        
        if len(histImage[xMin1:xMax1,yMin1:yMax1]) != 0:
            intersectionVoting[wndIdx] = sum(sum(histImage[int(np.floor(windows[wndIdx][0])):int(np.floor(windows[wndIdx][2])),int(np.floor(windows[wndIdx][1])):int(np.floor(windows[wndIdx][3]))]))
        else:
            intersectionVoting[wndIdx] = 0
    cv2.imwrite("histImage.jpg", histImage)
    cv2.imwrite("histImageSum.jpg", histImage + gray)
    
    return intersectionVoting

def detectVanishingPoints(intersectionVoting, imgOrig, windows):
    
    amIntVot = np.argmax(intersectionVoting)
    
    print(windows[amIntVot])
    win = windows[amIntVot]
    
    cpwind = imgOrig.copy()
    
    cpwind[int(win[0]):int(win[2]),int(win[1]):int(win[3]),0] = 255
    cpwind[int(win[0]):int(win[2]),int(win[1]):int(win[3]),1] = 255
    cpwind[int(win[0]):int(win[2]),int(win[1]):int(win[3]),2] = 255
    cv2.imwrite("bestWin.jpg", cpwind)
    
    return win


if __name__ == '__main__':
    
    img, gray, imgOrig = readImage('biblioteca.jpg')
    edges = preprocessing(gray)
    lines, data = detectLines(edges)
    nClusters = 10
    ret, label, center = clustering(data,nClusters)
    intersections = detectIntersections(nClusters, label, center, data, lines, img, imgOrig)
    windows = detectWindows(gray, intersections) 
    intersectionVoting = getIntersectionDescriptors(intersections, gray, windows)
    win = detectVanishingPoints(intersectionVoting, imgOrig, windows)
