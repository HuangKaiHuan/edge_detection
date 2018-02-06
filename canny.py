#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

__author__ = 'hkh'
__date__ = '05/02/2018'
__version__ = 1.0

"""
主要参考：　数字图像处理（第三版）10.2.6，Rafael C. Gonzalez
"""

import cv2
import numpy as np
import time


def geodesicDilation(src, mask, kernel, iteration=-1):
    if iteration > 0:
        while iteration:
            temp = cv2.dilate(src=src, kernel=kernel)
            src = cv2.min(src1=temp, src2=mask)
            iteration -= 1
        return src
    else:
        while True:
            temp = cv2.dilate(src=src, kernel=kernel)
            pre_image = src.copy()
            src = cv2.min(src1=temp, src2=mask)
            if 0 == cv2.compare(src1=src, src2=pre_image, cmpop=cv2.CMP_NE).sum():
                return src


def Canny_for_loop_imp(src, thresh1, thresh2):
    assert thresh1 < thresh2

    Gx = cv2.Sobel(src=src, ddepth=cv2.CV_32F, dx=1, dy=0)
    Gy = cv2.Sobel(src=src, ddepth=cv2.CV_32F, dx=0, dy=1)
    magnitude, angle = cv2.cartToPolar(Gx, Gy, angleInDegrees=True)

    rows, cols = src.shape
    suppression_edge = np.zeros_like(magnitude)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            mag = magnitude[i, j]
            ang = angle[i, j] % 180
            if mag >= thresh1:
                if ang < 22.5 or ang >= 157.5:
                    if mag >= magnitude[i, j + 1] and mag >= magnitude[i, j - 1]:
                        suppression_edge[i, j] = mag
                elif 67.5 > ang >= 22.5:
                    if mag >= magnitude[i - 1, j - 1] and mag >= magnitude[i + 1, j + 1]:
                        suppression_edge[i, j] = mag
                elif 112.5 > ang >= 67.5:
                    if mag >= magnitude[i - 1, j] and mag >= magnitude[i + 1, j]:
                        suppression_edge[i, j] = mag
                elif 157.5 > ang >= 112.5:
                    if mag >= magnitude[i - 1, j + 1] and mag >= magnitude[i + 1, j - 1]:
                        suppression_edge[i, j] = mag

    strong_edge = ((suppression_edge > thresh2) * 255).astype('uint8')
    full_edge = ((suppression_edge > thresh1) * 255).astype('uint8')
    optimized_edge = geodesicDilation(strong_edge, full_edge, kernel=np.ones((3, 3)))

    return cv2.convertScaleAbs(src=optimized_edge)


def Canny_maxtrix_parallel_imp(src, thresh1, thresh2):
    assert thresh1 < thresh2

    Gx = cv2.Sobel(src=src, ddepth=cv2.CV_32F, dx=1, dy=0)
    Gy = cv2.Sobel(src=src, ddepth=cv2.CV_32F, dx=0, dy=1)
    magnitude, angle = cv2.cartToPolar(Gx, Gy, angleInDegrees=True)

    angle %= 180
    angle_0_mask = (angle < 22.5) | (angle >= 157.5)
    angle_45_mask = (67.5 > angle) & (angle >= 22.5)
    angle_90_mask = (112.5 > angle) & (angle >= 67.5)
    angle_135_mask = (157.5 > angle) & (angle >= 112.5)

    dsize = (src.shape[1], src.shape[0])
    M = np.array([[1, 0, -1], [0, 1, 0]], dtype=np.float32)
    shift_left = cv2.warpAffine(magnitude, M, dsize)
    M = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    shift_right = cv2.warpAffine(magnitude, M, dsize)

    M = np.array([[1, 0,  0], [0, 1, -1]], dtype=np.float32)
    shift_up = cv2.warpAffine(magnitude, M, dsize)
    M = np.array([[1, 0,  0], [0, 1, 1]], dtype=np.float32)
    shift_down = cv2.warpAffine(magnitude, M, dsize)

    M = np.array([[1, 0,  1], [0, 1, 1]], dtype=np.float32)
    shift_right_down = cv2.warpAffine(magnitude, M, dsize)
    M = np.array([[1, 0,  -1], [0, 1, -1]], dtype=np.float32)
    shift_left_up = cv2.warpAffine(magnitude, M, dsize)

    M = np.array([[1, 0,  1], [0, 1, -1]], dtype=np.float32)
    shift_right_up = cv2.warpAffine(magnitude, M, dsize)
    M = np.array([[1, 0,  -1], [0, 1, 1]], dtype=np.float32)
    shift_left_down = cv2.warpAffine(magnitude, M, dsize)

    shift_left_right_max = cv2.max(shift_left, shift_right)
    shift_up_down_max = cv2.max(shift_up, shift_down)
    shift_rd_lu_max = cv2.max(shift_right_down, shift_left_up)
    shift_ru_lf_max = cv2.max(shift_right_up, shift_left_down)

    magnitude[angle_0_mask] *= (magnitude[angle_0_mask] >= shift_left_right_max[angle_0_mask])
    magnitude[angle_45_mask] *= (magnitude[angle_45_mask] >= shift_rd_lu_max[angle_45_mask])
    magnitude[angle_90_mask] *= (magnitude[angle_90_mask] >= shift_up_down_max[angle_90_mask])
    magnitude[angle_135_mask] *= (magnitude[angle_135_mask] >= shift_ru_lf_max[angle_135_mask])

    strong_edge = ((magnitude > thresh2) * 255).astype('uint8')
    full_edge = ((magnitude > thresh1) * 255).astype('uint8')
    optimized_edge = geodesicDilation(strong_edge, full_edge, kernel=np.ones((3, 3)))

    return cv2.convertScaleAbs(src=optimized_edge)


def Canny(src, thresh1, thresh2, imp='for_loop'):
    assert imp in ('for_loop', 'maxtrix_parallel', 'opencv')

    if 'for_loop' == imp:
        return Canny_for_loop_imp(src, thresh1, thresh2)
    if 'maxtrix_parallel' == imp:
        return Canny_maxtrix_parallel_imp(src, thresh1, thresh2)
    if 'opencv' == imp:
        return cv2.Canny(src, thresh1, thresh2, L2gradient=True)

if __name__ == '__main__':
    src = cv2.imread('./data/4.tif', 0)

    blur_img = cv2.GaussianBlur(src, ksize=(11, 11), sigmaX=0)
    t1 = time.time()
    res = Canny(src=blur_img, thresh1=20, thresh2=40, imp='maxtrix_parallel')
    t2 = time.time()
    res2 = Canny(src=blur_img, thresh1=20, thresh2=40, imp='opencv')
    t3 = time.time()
    print('python imp: ', t2 - t1)
    print('opencv imp: ', t3 - t2)

    cv2.namedWindow("src", cv2.WINDOW_NORMAL)
    cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    cv2.namedWindow("res2", cv2.WINDOW_NORMAL)
    cv2.imshow("src", src)
    cv2.imshow("res", res)
    cv2.imshow("res2", res2)
    cv2.waitKey()
