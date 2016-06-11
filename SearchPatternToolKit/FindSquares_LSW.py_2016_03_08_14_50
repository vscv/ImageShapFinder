#!/usr/bin/env python

'''
Simple "Square Detector" program.

Loads several images sequentially and tries to find squares in each image.
'''

# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3
print("Python version ", sys.version_info[0], ".", sys.version_info[1], ".", sys.version_info[2])

if PY3:
    xrange = range
"""
    >>> for i in range(0,255,26):
    ...     print(i)
    ...
    0
    26
    52
    78
    104
    130
    156
    182
    208
    234
    """

import numpy as np
import cv2
import math

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

"""
def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            
            cv2.imshow('bin', bin)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 5000 and cv2.contourArea(cnt) < 15000 and cv2.isContourConvex(cnt): #cv2.contourArea(cnt) < 8000
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares
    """
"""
    If you want to retrieve only the outter contours of the object, add the proper flag to
    cv2.findContours:
    contours,h = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
    otherwise it can return more than one contour per shape, hence the repeated result in
    the for loop.
    
    /Note/
    {mode}=={}
    提取模式.
    CV_RETR_EXTERNAL - 只提取最外層的輪廓
    CV_RETR_LIST - 提取所有輪廓，並且放置在 list 中
    CV_RETR_CCOMP - 提取所有輪廓，並且將其組織為兩層的 hierarchy: 頂層為連通域的外圍邊界，次層為洞的內層邊界。
    CV_RETR_TREE - 提取所有輪廓，並且重構嵌套輪廓的全部 hierarchy
    {method}
    逼近方法 (對所有節點, 不包括使用內部逼近的 CV_RETR_RUNS).
    CV_CHAIN_CODE - Freeman 鏈碼的輸出輪廓. 其它方法輸出多邊形(定點序列).
    CV_CHAIN_APPROX_NONE - 將所有點由鏈碼形式翻譯(轉化）為點序列形式
    CV_CHAIN_APPROX_SIMPLE - 壓縮水平、垂直和對角分割，即函數只保留末端的象素點;
    CV_CHAIN_APPROX_TC89_L1,
    CV_CHAIN_APPROX_TC89_KCOS - 應用 Teh-Chin 鏈逼近演算法. CV_LINK_RUNS - 通過連接為 1 的水平碎片使用完全不同的輪廓提取演算法。僅有 CV_RETR_LIST 提取模式可以在本方法中應用.
    {offset}
    每一個輪廓點的偏移量. 當輪廓是從圖像 ROI 中提取出來的時候，使用偏移量有用，因為可以從整個圖像上下文來對輪廓做分析.
    函數 cvFindContours 從二值圖像中提取輪廓，並且返回提取輪廓的數目。指針 first_contour 的內容由函數填寫。它包含第一個最外層輪廓的指針，如果指針為 NULL，則沒有檢測到輪廓（比如圖像是全黑的）。其它輪廓可以從 first_contour 利用 h_next 和 v_next 鏈接訪問到。 在 cvDrawContours 的樣例顯示如何使用輪廓來進行連通域的檢測。輪廓也可以用來做形狀分析和對象識別 - 見CVPR2001 教程中的 squares 樣例。該教程可以在 SourceForge 網站上找到。
    """

def find_squares(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## Need try more method for better result, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = cv2.bilateralFilter(img, 11, 17, 17)  # bilateralFilter is highly effective in noise removal while keeping edges sharp.
    img = cv2.medianBlur(img, 5)
    #kernel = np.ones((5,5),np.uint8)
    #img = cv2.dilate(img,kernel,iterations = 1)
    #img = cv2.erode(img,kernel,iterations = 1)
    #img = cv2.Canny(img, 30, 200)
    cv2.imshow('filter of img', img)
    
    squares = []
    for gray in cv2.split(img):                 # why split to 3 color-channel? And range() gray interval it looks like multiple image can found more cnts but some cnt will be repeated in diff color-thrs-image.
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)     # color camera never come to this line!!
            
            cv2.imshow('filter of bin', bin)
            bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            #cv2.imshow('filter of bin', bin)
            
            i = 0
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.015*cnt_len, True)
                
                if len(cnt) == 4 and cv2.contourArea(cnt) > 5000 and cv2.contourArea(cnt) < 15000 and cv2.isContourConvex(cnt): #cv2.contourArea(cnt) < 8000
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    
                    if max_cos < 0.1:
                        squares.append(cnt)
                        
                        # Find center and theta #
                        M = cv2.moments(cnt)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        #print(M)
                        #area = cv2.contourArea(cnt)
                        #perimeter = cv2.arcLength(cnt,True)
                        
                        # Find center and theta #
                        rows,cols = img.shape[:2]
                        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
                        angle = math.atan(vy/vx)
                        #print("angle radians = ", angle)
                        angle = math.degrees(angle)
                        #print("angle degrees = ", angle)
                        #print("Vxyz = \n", vx,vy,x,y, "angle of central line = \n", angle)
                        lefty = int((-x*vy/vx) + y)
                        righty = int(((cols-x)*vy/vx)+y)
                        cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
                        
                        # print all #
                        #print("centroid x,y = ", cx, cy, "area = : ", area, "contour length = ",  perimeter, "\n")
                        print("<S", i+1, "> x", cx, "y", cy, "theta", angle)
                        i = i + 1


    return squares




    #   \LSW using only gray channel and convert to binary image for findcontours    #

def find_squares_LSW(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## Need try more method for better result, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ##img = cv2.bilateralFilter(img, 11, 17, 17)  # bilateralFilter is highly effective in noise removal while keeping edges sharp.
    ##img = cv2.medianBlur(img, 5)
    #kernel = np.ones((5,5),np.uint8)
    #img = cv2.dilate(img,kernel,iterations = 1)
    #img = cv2.erode(img,kernel,iterations = 1)
    #img = cv2.Canny(img, 30, 200)
    cv2.imshow('filter of img', img)
    
    squares = []
    #bin = cv2.Canny(img, 0, 50, apertureSize=5)
    #bin = cv2.dilate(bin, None)
    #cv2.imshow('filter of canny', bin)
    
    #
    # Thresholding to bainary image
    #
    
    #thrs = 160                                  # 0->255=black->white, more white (high reflection) obj will be found.
    #retval, bin = cv2.threshold(img, thrs, 255, cv2.THRESH_BINARY)
    #
    
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #cv2.imshow("hist of bin", plt.hist(blur.ravel(),256))
    #ret3,bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #
    
    # cv2.ADAPTIVE_THRESH_MEAN_C : threshold value is the mean of neighbourhood area.
    #bin = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #
    
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
    bin = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #
    
    # Show thresholded image
    cv2.imshow('Threshold of img', img)
    #
    
    # FindContours()                                    # RETR_CCOMP RETR_TREE RETR_EXTERNAL RETR_LIST
    bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow('Contours of img', bin)
    
    i = 0
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.015*cnt_len, True)
        
        if len(cnt) == 4 and cv2.contourArea(cnt) > 5000 and cv2.contourArea(cnt) < 15000 and cv2.isContourConvex(cnt): #cv2.contourArea(cnt) < 8000
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
            
            if max_cos < 0.1:
                squares.append(cnt)
                
                # Find center and theta #
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                #print(M)
                #area = cv2.contourArea(cnt)
                #perimeter = cv2.arcLength(cnt,True)
                
                
                # Find center and theta #
                rows,cols = img.shape[:2]
                [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
                angle = math.atan(vy/vx)
                #print("angle radians = ", angle)
                angle = math.degrees(angle)
                #print("angle degrees = ", angle)
                #print("Vxyz = \n", vx,vy,x,y, "angle of central line = \n", angle)
                ##lefty = int((-x*vy/vx) + y)
                ##righty = int(((cols-x)*vy/vx)+y)
                ##cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
                
                # print all #
                #print("centroid x,y = ", cx, cy, "area = : ", area, "contour length = ",  perimeter, "\n")
                print("<S", i+1, "> x", cx, "y", cy, "theta", angle)
                i = i + 1


    if len(squares) <= 0:
        #break!!!!
        print('No object found!!! Please fill/move the plate.')
        print(len(squares))
    
    return squares




def FindSquare_from_SingleImage(find_squares):
    from glob import glob
    #for fn in glob('../data/pic*.png'):#20160201-150810.ppm
    for fn in glob('20160205-170407.jpg'):#20160201-150810.ppm
        img = cv2.imread(fn)
        
        #
        squares = find_squares(img)
        cv2.drawContours( img, squares, -1, (0, 255, 0), 3 )        # 0 for now cnt[0], -1 for all cnt, 1-n for cnt[1:n]
        #cv2.imshow('squares from' + fn, img)
        
        
        # PutText (x,y, theta)
        
        
        # copy a small image for demo record #
        smallimg = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('squares from ' + fn, smallimg)
        
        #a, b, c = squares.shape            # AttributeError:'list' object has no attribute 'shape'
        #print(range(len(squares)))          # range(0,6)
        #print(squares)
        
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
    cv2.destroyAllWindows()



def DEMO_FindSquare_from_VideoCamera(find_squares):
    print("OK! GO!!")
    import numpy as np
    import cv2
    import time

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 1)        # not works!!!
    #cv2.SetCaptureProperty(cap,cv.CV_CAP_PROP_FRAME_WIDTH,1280)      # Add for logitec cam
    #cv2.SetCaptureProperty(cap,cv.CV_CAP_PROP_FRAME_HEIGHT,720)
    #cap.set(3,1280)
    #cap.set(4,720)
    #cap.set(3,1920)
    #cap.set(4,1080)

    while(cap.isOpened()):
        ret, frame = cap.read()
        #print(frame.shape)     #(720, 1280, 3)
    
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = frame
    
        #cv2.imshow(' [ ***** USB Cam Image ***** ] ',frame)
        #cv2.imshow(' [ ***** USB Cam Image ***** ] ',smallimg)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            #If the FPS is equal to 20, then you should wait 0,05 seconds between the consecutive frames. So put waitKey(50) after imshow() and it will be displayed at normal speed. # 1000/FPS = waitkey_time, 1000/30 = waitkey(33)
            #timename = time.strftime("%Y%m%d-%H%M%S")
            #timename = 'TEST'
            #imgname = timename + '.ppm'
            #cv2.imwrite(timename+".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) #OK
            #cv2.imwrite(imgname, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) #OK
            #cv2.imwrite(timename+".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            break
        
        # add FindSquare here #
        #img = cv2.imread(frame)
        img = frame
        squares = find_squares(img)
        cv2.drawContours( img, squares, -1, (0, 255, 0), 3 )        # 0 for now cnt[0], -1 for all cnt, 1-n for cnt[1:n]

        #cv2.imshow('squares from' + fn, img)
        
        # copy a small image for demo record #
        smallimg = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        #cv2.imshow('squares from ' + fn, smallimg)
        
        #a, b, c = squares.shape            # AttributeError:'list' object has no attribute 'shape'
        #print(range(len(squares)))          # range(0,6)
        #print(squares)

        cv2.imshow(' [ ***** USB Cam Image ***** ] ',smallimg)
    
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    #FindSquare_from_SingleImage(find_squares_LSW)
    #FindSquare_from_SingleImage(find_squares)
    DEMO_FindSquare_from_VideoCamera(find_squares_LSW)
    #DEMO_FindSquare_from_VideoCamera(find_squares)





