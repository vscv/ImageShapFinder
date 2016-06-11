#!/usr/bin/env python3
#
# By Shi-Wei Lo, NCHC, 2016/03/08.
# LSW@nchc.org.tw
#

from SearchPatternToolKit import SearchPatternToolKit as TK
from SearchPatternToolKit import SearchCircles as SC

def man():

#   Print afix information   #
    TK.ThisWorks()
    TK.ShowPyOcvOSVersion()
    
#   Run Circle Search program   #
    TK.PrintNowSC()
    SC.GetCirclesFromCamera()

#   Demo Circle Search program   #
    #TK.DEMO_FindCircle_from_VideoCamera()

if __name__ == '__main__':
    man()
