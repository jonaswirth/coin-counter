import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
from numpy import unravel_index
import math
from sklearn.cluster import KMeans

DEBUG = True
IMG_HEIGHT = 720
IMG_WIDTH = 1280

def main():
    print("..")
    #cv.namedWindow("cam", cv.WINDOW_NORMAL)
    cap = cv.VideoCapture(1, cv.CAP_DSHOW)
    ret = cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    success, img = cap.read()
    plt.imshow(img)
    cv.imwrite("shot1.png", img)
    plt.show()

def detect_coins(img):
    return

# def detect_corners(img):
#     corners = cv.cornerHarris(img, 3, 3, 0.04)
#     max_indexes = np.argpartition(corners.flatten(), -4)[-4:]
#     return unravel_index(max_indexes, img.shape)

def show_and_save_image_debug(step:int, img, cmap = "gray"):
    plt.imshow(img, cmap=cmap)
    plt.show()
    cv.imwrite(f"debug/step{step}.png", img)

def print_hough_lines(shape, lines):
    cdst = np.zeros(shape)

    for i in range(0, len(lines)):
        rho = lines[i][0]
        theta = lines[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1280*(-b)), int(y0 + 1280*(a)))
        pt2 = (int(x0 - 1280*(-b)), int(y0 - 1280*(a)))
        cv.line(cdst, pt1, pt2, 255, 3, cv.LINE_AA)
    
    return cdst




def detect_corners(img):
    canny_img = cv.Canny(img, 50, 200, None, 3) #TODO: check these params

    if DEBUG:
        show_and_save_image_debug(1, canny_img)

    lines = cv.HoughLines(canny_img, 1, np.pi / 180, 150)
    mapped_lines = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        mapped_lines.append([rho, theta])
    
    print(mapped_lines)
    lines = mapped_lines
    
    if DEBUG:
        printed_lines = print_hough_lines(canny_img.shape, lines)
        show_and_save_image_debug(2, printed_lines)
    
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(mapped_lines)

    clustered_Lines = kmeans.cluster_centers_

    if DEBUG:
        printed_lines = print_hough_lines(canny_img.shape, clustered_Lines)
        show_and_save_image_debug(3, printed_lines)


    return lines


img = plt.imread("static.png") ##Todo replace with live image
img = np.array(cv.cvtColor(img, cv.COLOR_BGR2GRAY) * 255, dtype="uint8")

lines = detect_corners(img)
print(lines)

plt.imshow(img, cmap="gray")
plt.show()