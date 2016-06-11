#!/usr/bin/env python3
#
# By Shi-Wei Lo, NCHC, 2016/03/08.
# LSW@nchc.org.tw
#

from SearchPatternToolKit import SearchPatternToolKit as TK
from SearchPatternToolKit import SearchRectangles as SR


def man():

#   Print afix information   #
    TK.ThisWorks()
    TK.ShowPyOcvOSVersion()
    
#   Run Circle Search program   #
    TK.PrintNowSR()
    #SC.GetRectanglesFromCamera()
    SR.GetRectangesFromCamera(SR.find_squares_LSW)

#   Demo rectangle Search program   #
    #TK.DEMO_FindCircle_from_VideoCamera()

if __name__ == '__main__':
    man()
