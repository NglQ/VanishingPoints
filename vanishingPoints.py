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
    #6) Use dynamic programming to find the intersections
    #7) Find a way to detect more than one Vanishing point

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

def getLineParameters(ln):
    m = (ln[3] - ln[1]) / (ln[2] - ln[0])
    q = ln[1] - m * ln[0]
    return m,q

def intersect(q1, q2, m1, m2):
    xint = (q1-q2) / (m2-m1)
    yint = m1 * xint + q1
    return xint, yint

def detectIntersections(lines, img, imgOrig):
    intersections = []
    
    #print(lines)
    
    for i in range(len(lines)):
        ln1 = lines[i][0]
        #print(lines[i])
        if ln1[2] - ln1[0] != 0.0:            
            m1, q1 = getLineParameters(ln1)
            
            if math.sin(m1) < 0.984807753 and math.sin(m1) > -0.984807753:
            
                for j in range(i,len(lines)):
                    ln2 = lines[j][0]
                    
                    if ln2[2] - ln2[0] != 0.0:
                    
                        m2, q2 = getLineParameters(ln2)
                        
                        if m2-m1 != 0.0 and math.sin(m2) < 0.984807753 and math.sin(m2) > -0.984807753:                       
                            xint, yint = intersect(q1, q2, m1, m2)
                        else:
                            xint = np.inf
                            yint = np.inf
                        if not math.isnan(xint) and not math.isnan(yint) and not math.isinf(xint) and not math.isinf(yint):
                            img = cv2.circle(img, (round(xint),round(yint)), radius = 1, color = (255,0,0), thickness = -1)
                            intersections.append(np.array([xint, yint]))
            
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
        
        if len(histImage[yMin1:yMax1,xMin1:xMax1]) != 0:
            intersectionVoting[wndIdx] = sum(sum(histImage[yMin1:yMax1,xMin1:xMax1]))
        else:
            intersectionVoting[wndIdx] = 0
    cv2.imwrite("histImage.jpg", histImage)
    cv2.imwrite("histImageSum.jpg", histImage + gray)
    
    return intersectionVoting

def detectVanishingAreas(intersectionVoting, imgOrig, windows):
    
    #---------- new function ---------------------
    winBest = []
    maxVote = max(intersectionVoting)
    maxValuesIdxs = intersectionVoting == maxVote
    #print(intersectionVoting[maxValuesIdxs])
    
    windowsNp = np.array(windows)
    
    #print(windowsNp[maxValuesIdxs])
    
    winIdxMax = windowsNp[maxValuesIdxs]
    
    cpwind = imgOrig.copy()
    
    for win in winIdxMax:
        
        cpwind[int(win[1]):int(win[3]),int(win[0]):int(win[2]),0] = 255
        cpwind[int(win[1]):int(win[3]),int(win[0]):int(win[2]),1] = 0
        cpwind[int(win[1]):int(win[3]),int(win[0]):int(win[2]),2] = 0
    
    cv2.imwrite("bestWin.jpg", cpwind)
    #------------------------------------------
    
    return winBest


if __name__ == '__main__':
    
    img, gray, imgOrig = readImage('twoVanPnts.jpg')
    edges = preprocessing(gray)
    lines, data = detectLines(edges)
    intersections = detectIntersections(lines, img, imgOrig)
    windows = detectWindows(gray, intersections) 
    intersectionVoting = getIntersectionDescriptors(intersections, gray, windows)
    win = detectVanishingAreas(intersectionVoting, imgOrig, windows)
