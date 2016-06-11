#!/usr/bin/env python3


# Find circles' center and area
# [ with Py3.5.1, CV3.1.0 ]
#





'''
This example illustrates how to use cv2.HoughCircles() function.

Usage:
    houghcircles.py [<image_name>]
    image argument defaults to ../data/board.jpg
'''

"""
    Finds circles in a grayscale image using the Hough transform.
    
    C++: void HoughCircles(InputArray image, OutputArray circles, int method, double dp, double minDist, double param1=100, double param2=100, int minRadius=0, int maxRadius=0 )
    C: CvSeq* cvHoughCircles(CvArr* image, void* circle_storage, int method, double dp, double min_dist, double param1=100, double param2=100, int min_radius=0, int max_radius=0 )
    Python: cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) → circles
    
    Parameters:
    image – 8-bit, single-channel, grayscale input image.
    circles – Output vector of found circles. Each vector is encoded as a 3-element floating-point vector  (x, y, radius) .
    circle_storage – In C function this is a memory storage that will contain the output sequence of found circles.
    method – Detection method to use. Currently, the only implemented method is CV_HOUGH_GRADIENT , which is basically 21HT , described in [Yuen90].
    dp – Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
    minDist – Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    param1 – First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
    param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
    minRadius – Minimum circle radius.
    maxRadius – Maximum circle radius.
    """

# Python 2/3 compatibility
#from __future__ import print_function   # let 3.x using the "print"

import cv2
import numpy as np
import sys


""" comment for demoe record
if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except IndexError:
        fn = "../data/board.jpg"

    src = cv2.imread(fn, 1)
    imgG = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    imgF = cv2.medianBlur(imgG, 5)
    kernel = np.ones((5,5),np.uint8)
    imgF = cv2.dilate(imgF,kernel,iterations = 1)
    imgF = cv2.erode(imgF,kernel,iterations = 1)


    cimg = src.copy() # numpy function


    # For 640x426, cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) ＝(imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
    #circles = cv2.HoughCircles(imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)#circles	Output vector of found circles. Each vector is encoded as a 3-element floating-point vector (x,y,radius)

    # For 1600x1200 (imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 45, 50)
    circles = cv2.HoughCircles(imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 45, 50)#circles	Output vector of found circles. Each vector is encoded as a 3-element floating-point vector (x,y,radius)


    a, b, c = circles.shape         # 1, 12, 3



    for i in range(b):
        cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv2.LINE_AA) # B G R
        cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), 1, (255, 0, 255), 2, cv2.LINE_AA)  # draw center of circle
        #line((screenWidth/2)-20, screenHeight/2, (screenWidth/2)+20, screenHeight/2);  //crosshair horizontal
        #line(screenWidth/2, (screenHeight/2)-20, screenWidth/2, (screenHeight/2)+20);  //crosshair vertical # draw center of circle

        print("<C", i+1, "> x", circles[0][i][0], "y", circles[0][i][1], "radius", circles[0][i][2])
        #print(a,b,c)
        #print(circles[0][i])
        #sprintf(R, "%f", circles[0][i][0]);        # for c/c++ type
        #(circles[0][i][0])
        
        ## Texting the X, Y, theta.
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cimg, str(circles[0][i][0])+ ", "+ str(circles[0][i][1])+ ", "+ str(circles[0][i][2]), (circles[0][i][0], circles[0][i][1]), font, 0.3,(255,255,255),1,cv2.LINE_AA)
    
    cv2.imshow("source", src)
    cv2.imshow("Gray", imgG)
    cv2.imshow("filtering", imgF)
    cv2.imshow("detected circles", cimg)
    cv2.waitKey(0)
"""




"""
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
    param1=50,param2=30,minRadius=0,maxRadius=0)
    
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    """


#"""
##  This only for DEMO video    ##
def DEMO_FindCircle_from_VideoCamera():
    import cv2
    import numpy as np
    import sys
    from SearchPatternToolKit import SearchPatternToolKit as TK
        
        """ HoughCircles Parameters:
            
            cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) ＝(imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
            
            circles = Output vector of found circles. Each vector is encoded as a 3-element floating-point vector (x,y,radius)
            
            Para:
            image – 8-bit, single-channel, grayscale input image.
            circles – Output vector of found circles. Each vector is encoded as a 3-element floating-point vector  (x, y, radius) .
            circle_storage – In C function this is a memory storage that will contain the output sequence of found circles.
            method – Detection method to use. Currently, the only implemented method is CV_HOUGH_GRADIENT , which is basically 21HT , described in [Yuen90].
            dp – Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height. (set to 1)
            minDist – Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed. (set to 10)
            param1 – First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller). (set to 100)
            param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first. (set to 30)
            minRadius – Minimum circle radius. (set to 45)
            maxRadius – Maximum circle radius.(set to 50)"""
    
    # To Get the paramenters from _CONF file #
    config = TK.Config.GetConf()
    CircleImgFilter = TK.Config.GetCircleFilter(config)
    print("--CircleImgFilter = ", CircleImgFilter)
    #quit()

    cap = cv2.VideoCapture(1)
    #cap.set(cv2.CAP_PROP_FPS, 1)        # not works!!!
    #cv2.SetCaptureProperty(cap,cv.CV_CAP_PROP_FRAME_WIDTH,1280)      # Add for logitec cam
    #cv2.SetCaptureProperty(cap,cv.CV_CAP_PROP_FRAME_HEIGHT,720)
    #cap.set(3,1280)
    #cap.set(4,720)
    #cap.set(3,1920)
    #cap.set(4,1080)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        
        #src = cv2.imread(fn, 1)
        src = frame
        imgG = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        if CircleImgFilter == "None":
            
            imgF = imgG
        
        elif CircleImgFilter == "Median":
            
            imgF = cv2.medianBlur(imgG, 5)
        
        elif CircleImgFilter == "Median+dilate":
            
            imgF = cv2.medianBlur(imgG, 5)
            kernel = np.ones((5,5),np.uint8)
            imgF = cv2.dilate(imgF,kernel,iterations = 1)
        
        elif CircleImgFilter == "Median+dilate+erode":
            
            imgF = cv2.medianBlur(imgG, 5)
            kernel = np.ones((5,5),np.uint8)
            imgF = cv2.dilate(imgF,kernel,iterations = 1)
            imgF = cv2.erode(imgF,kernel,iterations = 1)

        #   Check current CircleImgFilter   #
        #print("--CircleImgFilter = ", CircleImgFilter)

        #imgF = imgG
        #imgF = cv2.medianBlur(imgG, 5)
        #kernel = np.ones((5,5),np.uint8)
        #imgF = cv2.dilate(imgF,kernel,iterations = 1)
        # imgF = cv2.erode(imgF,kernel,iterations = 1)

        #cv2.imshow("source", src)
        
        
        cimg = src.copy() # numpy function
        # For 640x426,   (imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
        #circles = cv2.HoughCircles(imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
        #For 1600x1200, (imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 45, 50)
        circles = cv2.HoughCircles(imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 45, 50)

        
        #a, b, c = circles.shape         # 1, 12, 3
        
        
        if circles is None:
            print('No object found!!! Please fill/move the plate.')
        else:
            
            #print('We Got the CIRCLE !!!!!!!!')
            
            print(len(circles))
            a, b, c = circles.shape
            for i in range(b):
                cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv2.LINE_AA) # B G R
                cv2.circle(cimg, (circles[0][i][0], circles[0][i][1]), 1, (255, 0, 255), 2, cv2.LINE_AA)  # draw center of circle
                #line((screenWidth/2)-20, screenHeight/2, (screenWidth/2)+20, screenHeight/2);  //crosshair horizontal
                #line(screenWidth/2, (screenHeight/2)-20, screenWidth/2, (screenHeight/2)+20);  //crosshair vertical # draw center of circle
        
                print("<C", i+1, "> x", circles[0][i][0], "y", circles[0][i][1], "radius", circles[0][i][2])
                #print(a,b,c)
                #print(circles[0][i])
                #sprintf(R, "%f", circles[0][i][0]);        # for c/c++ type
                #(circles[0][i][0])
        
                ## Texting the X, Y, theta.
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cimg, str(circles[0][i][0])+ ", "+ str(circles[0][i][1])+ ", "+ str(circles[0][i][2]), (circles[0][i][0], circles[0][i][1]), font, 0.3,(255,255,255),1,cv2.LINE_AA)
                    
                

        #   Check image's loading, color->gray and filtered result  #
        #cv2.imshow("source", src)
        #cv2.imshow("Gray", imgG)
        #cv2.imshow("filtering", imgF)

        #   BC's 16200*1200 may large than your LCD screen size, so it may dispay half size of image    #
        smallimg = cv2.resize(cimg, (0,0), fx=0.5, fy=0.5)
        cv2.imshow("detected circles", smallimg)

        #   Setint FPS by waitkey(N), N = # of usec, and quit the program by press 'q' key    #
        #cv2.waitKey(0)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    DEMO_FindCircle_from_VideoCamera()
#"""

