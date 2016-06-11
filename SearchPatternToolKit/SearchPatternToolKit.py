#!/usr/bin/env python3
#
# By Shi-Wei Lo, NCHC, 2016/03/08.
# LSW@nchc.org.tw
#
"""SearchPatternToolKit for import function modules.
    
    This Search Pattern Tool Kit provides two main pattern search functions.
    1. [FindCircle] :
    2. [FindRectangle] :
    3. and a few functions for camera pre-viewing, save current image, parsing parameter and so on.
    """


#   Print the prgram information   #

def ThisWorks():
    from SearchPatternToolKit import CharGrid as CG
    print()
    CG.resize(8,50)
    CG.add_rectangle(0, 0, 8, 50, "|") # (0, 0, 8, 50, "%", fill=True)
    CG.add_text(2, 14, "[SearchPatternToolKit]")
    CG.add_text(3, 20, "Version 0.1")
    CG.add_text(4, 20, "2016/03/09")
    CG.add_text(5, 20, "LSW, NCHC.")
    CG.render(False)
    print()

def ShowPyOcvOSVersion():
    import os
    import sys
    import platform
    import cv2
    #print("--Runing", os.name)
    #print("--OS is", sys.platform)
    #print("--OS platform is", platform.machine())
    print("--Python version = ", sys.version_info[0], ".", sys.version_info[1], ".", sys.version_info[2], sep='')
    print("--OpencV version =", cv2.__version__)
    print("--OS platform =", os.name, ",", sys.platform, ",", platform.machine())

def PrintNowSC():
    print("--")
    print("--Start Searching Cicles...")
    print("--")

def PrintNowSR():
    print("--")
    print("--Start Searching Rectangles...")
    print("--")

import configparser
import os
class Config:

    # test parser method
    def GetConfig_test(config):
        
        # Check reading parameters
        print(config.sections())
        print("DEFAULT Img W =", config['DEFAULT']['Img_Size_W'])
    
        W = config['DEFAULT']['Img_Size_W']
        H = config['DEFAULT']['Img_Size_H']
        return W,H

    # read and return config
    def GetConf():
        if os.path.isfile('_Conf'):
            print("--Configure reading...")
        else:
            print("--Configure file not found!")
            print("--Please check '_Conf' is exist.")
            quit("Quit the program!")
    
        config = configparser.ConfigParser()
        config.read('_Conf')
        return config

    # DEFAULT parameters #
    def GetDEFAULTFPS(config):
        DEFAULTFPS = config['DEFAULT']['影像擷取頻率參數']
        return DEFAULTFPS
    
    def GetDEFAULTImgScale(config):
        DEFAULTImgScale = config['DEFAULT']['畫面縮放參數']
        return DEFAULTImgScale
    
    # Circle parameters #

    def GetCircleRadiusMax(config):
        CircleRadius = config['圓形開放參數表']['圓形像素半徑參數上限']
        return CircleRadius
    
    def GetCircleRadiusMin(config):
        CircleRadius = config['圓形開放參數表']['圓形像素半徑參數下限']
        return CircleRadius
    
    def GetCircleFilter(config):
        CircleFilter = config['圓形開放參數表']['圓形影像過濾參數']
        return CircleFilter


    # Rectangle parameters #

    def GetRectanglePixelSizeMax(config):
        RectanglePixelSize = config['矩形開放參數表']['矩形像素面積參數上限']
        return RectanglePixelSize

                              
    def GetRectanglePixelSizeMin(config):
        RectanglePixelSize = config['矩形開放參數表']['矩形像素面積參數下限']
        return RectanglePixelSize


    def GetRectangleFilter(config):
        RectangleFilter = config['矩形開放參數表']['矩形影像過濾參數']
        return RectangleFilter


# Read configure and print out
import configparser
def PrintConf():
    config = configparser.ConfigParser()
    config.read('_Conf')
    HH = TK.Config.GetConfig(config)
    print("W,H is from class mode>>>>", HH)
    
    RPSMax = TK.Config.GetRectanglePixelSizeMax(config)
    print("RPS is from class mode>>>>", RPSMax)
    RPSMin = TK.Config.GetRectanglePixelSizeMin(config)
    print("RPS is from class mode>>>>", RPSMin)
    
    if TK.Config.GetCircleFilter(config) == "MD5":
        print("TEST filter = true")
    else:
        print("TEST filter = others")


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
    
    # Get paramenters from _CONF file #
    # Read parameters
    config = TK.Config.GetConf()
    # FPS
    FPS = int(TK.Config.GetDEFAULTFPS(config))
    # Scale
    ImgScale = float(TK.Config.GetDEFAULTImgScale(config))
    print("--FPS = ", FPS, ", ImgScale= ", ImgScale)
    # Radious Max & Min
    maxRadius = int(TK.Config.GetCircleRadiusMax(config))
    minRadius = int(TK.Config.GetCircleRadiusMin(config))
    print("--Radius Max =", maxRadius, ", Radius Min =", minRadius)
    # Filter
    CircleImgFilter = TK.Config.GetCircleFilter(config)
    print("--CircleImgFilter = ", CircleImgFilter)
    #quit()
    
    cap = cv2.VideoCapture(-1)
    #cap.set(cv2.CAP_PROP_FPS, 1)        # not works for logitec cam (不支援OS X系統)!!!
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
        
        # Default image filter
        #imgF = imgG
        #imgF = cv2.medianBlur(imgG, 5)
        #kernel = np.ones((5,5),np.uint8)
        #imgF = cv2.dilate(imgF,kernel,iterations = 1)
        # imgF = cv2.erode(imgF,kernel,iterations = 1)
        
        #cv2.imshow("source", src)
        
        # Temp image
        cimg = src.copy() # numpy function
        
        # For 640x426,   (imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
        #circles = cv2.HoughCircles(imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
        # For 1600x1200, (imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 45, 50)
        circles = cv2.HoughCircles(imgF, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, minRadius, maxRadius)
        
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

        #   BC's of 1600*1200 may large than your LCD screen size, so it may dispay half size of image    #
        # ImgScale to scaling image size, 0.5 = half
        smallimg = cv2.resize(cimg, (0,0), fx = ImgScale, fy = ImgScale)
        cv2.imshow("== Search circles ==", smallimg)
        
        #   Setint FPS by waitkey(N), N = # of usec, and quit the program by press 'q' key    #
        # FPS to frame rate per sec
        
        if cv2.waitKey(FPS) & 0xFF == ord('q'):
            break
        # read most frames as you can
        #cv2.waitKey(0)



