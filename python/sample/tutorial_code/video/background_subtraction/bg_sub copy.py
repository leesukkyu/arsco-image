from __future__ import print_function
import cv2 as cv
import argparse


backSub = cv.createBackgroundSubtractorMOG2()
# backSub = cv.createBackgroundSubtractorKNN()

# [capture]
capture = cv.VideoCapture(cv.samples.findFileOrKeep(
    '/Users/user/Desktop/git/arsco-image/sample/tutorial_code/video/background_subtraction/vtest.avi'))
background = cv.imread(
    '/Users/user/Desktop/git/arsco-image/sample/tutorial_code/video/background_subtraction/background.jpeg')
current = cv.imread(
    '/Users/user/Desktop/git/arsco-image/sample/tutorial_code/video/background_subtraction/current.jpeg')
background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
current = cv.cvtColor(current, cv.COLOR_BGR2GRAY)
# [capture]

fgMask = backSub.apply(background)
fgMask = backSub.apply(current)
cv.imshow('FG Mask', fgMask)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # [apply]
    # update the background model

    # [show]
    # show the current frame and the fg masks
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
