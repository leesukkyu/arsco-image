from __future__ import print_function
import cv2 as cv
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

# [설정]
ALGORITHM = 'MOG2'  # 알고리즘 선택 MOG2 | KNN
BACKGROUND_PATH = '/Users/user/Desktop/git/arsco-image/python/src/backgroundSubtractor/1.jpg'  # 기본 이미지
FOREGROUND_PATH = '/Users/user/Desktop/git/arsco-image/python/src/backgroundSubtractor/2.jpg'  # 기본 이미지


def grayScale(img):  # 흑백 필터
    result = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return result


def median(img):  # 블러 필터
    result = cv.medianBlur(img, 5)
    return result


def threshold(img):  # 이진화를 통한, 임계값 처리 # parameters img, 임계값, 변환값, 이진화 옵션
    # ret, result = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
    # ret, result = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    ret, result = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
    return result


def morphology(img):  # 수축 팽창 처리
    kernel = np.ones((10, 10), np.uint8)
    # result = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    result = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return result


def edge(img):  # 라인 처리
    return cv.Canny(img, 100, 200)


def imageProcessing(img):  # 이미지 처리
    img = grayScale(img)
    img = threshold(img)
    img = morphology(img)
    # img = edge(img)
    # img = median(img)
    return img


def detectObject(img):  # 결과물에 오브젝트가 있는지 검사
    contours, hierarchy = cv.findContours(
        img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    result = cv.imread(FOREGROUND_PATH)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w > 80:
            cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("2", result)


def init():  # 이미지 비디오 분석
    if ALGORITHM == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN()

    background = cv.imread(BACKGROUND_PATH)  # 백그라운드 이미지 로드
    background = imageProcessing(background)  # 프로세싱

    foreground = cv.imread(FOREGROUND_PATH)  # 포어그라운드 이미지 로드
    foreground = imageProcessing(foreground)  # 프로세싱

    # 이미지 비교
    backSub.apply(background)
    mask = backSub.apply(foreground)

    # 보기
    cv.imshow('Mask', mask)
    detectObject(mask)
    cv.waitKey(100000)


def init2():  # 웹캠으로 비디오 분석
    if ALGORITHM == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN()

    capture = cv.VideoCapture(0)

    if (capture.isOpened() == False):
        print("Unable to read camera feed")
        exit(0)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        mask = backSub.apply(imageProcessing(frame))  # 이미지 비교

        # 보기
        cv.imshow('Frame', frame)
        cv.imshow('Mask', mask)
        detectObject(mask)

        keyboard = cv.waitKey(30)  # 프레임 조절
        if keyboard == 'q' or keyboard == 27:
            break


init()
