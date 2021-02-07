from __future__ import print_function
import cv2 as cv
import argparse


backSub = cv.createBackgroundSubtractorMOG2()
# backSub = cv.createBackgroundSubtractorKNN()

# [capture]
capture = cv.VideoCapture(cv.samples.findFileOrKeep(
    '/Users/user/Desktop/git/arsco-image/sample/tutorial_code/video/background_subtraction/vtest.avi'))
if not capture.isOpened():
    print('Unable to open')
    exit(0)
# [capture]

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # [apply]
    # update the background model
    fgMask = backSub.apply(frame)

    # [show]
    # show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
