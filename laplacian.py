#!/usr/bin/python3.4
# -*- coding:utf-8 -*-

__author__ = 'hkh'
__date__ = '31/01/2018'
__version__ = 1.0

import cv2
import numpy as np

if __name__ == '__main__':
    src = cv2.imread('./data/3.tif', 0)
    cv2.imshow('src', src)

    kernel = np.array([[-1, -1, -1],
                       [2, 2, 2],
                       [-1, -1, -1]])
    horizontal_edge = cv2.filter2D(src, cv2.CV_32F, kernel)
    horizontal_edge = cv2.convertScaleAbs(horizontal_edge)
    # _, horizontal_edge = cv2.threshold(horizontal_edge, horizontal_edge.max() * 0.8, 255, cv2.THRESH_BINARY)
    cv2.imshow('horizontal_edge', horizontal_edge)

    kernel = np.array([[-1, 2, -1],
                       [-1, 2, -1],
                       [-1, 2, -1]])
    vertical_edge = cv2.filter2D(src, ddepth=cv2.CV_32F, kernel=kernel)
    vertical_edge = cv2.convertScaleAbs(vertical_edge)
    # _, vertical_edge = cv2.threshold(vertical_edge, vertical_edge.max() * 0.8, 255, cv2.THRESH_BINARY)
    cv2.imshow('vertical_edge', vertical_edge)

    kernel = np.array([[-1, -1, 2],
                       [-1, 2, -1],
                       [2, -1, -1]])
    positive_45_deg_edge = cv2.filter2D(src, ddepth=cv2.CV_32F, kernel=kernel)
    positive_45_deg_edge = cv2.convertScaleAbs(positive_45_deg_edge)
    # _, positive_45_deg_edge = cv2.threshold(positive_45_deg_edge, positive_45_deg_edge.max() * 0.8, 255, cv2.THRESH_BINARY)
    cv2.imshow('positive_45_deg_edge', positive_45_deg_edge)

    kernel = np.array([[2, -1, -1],
                       [-1, 2, -1],
                       [-1, -1, 2]])
    negative_45_deg_edge = cv2.filter2D(src, ddepth=cv2.CV_32F, kernel=kernel)
    negative_45_deg_edge = cv2.convertScaleAbs(negative_45_deg_edge)
    # _, negative_45_deg_edge = cv2.threshold(negative_45_deg_edge, negative_45_deg_edge.max() * 0.8, 255, cv2.THRESH_BINARY)
    cv2.imshow('negative_45_deg_edge', negative_45_deg_edge)

    cv2.waitKey()
