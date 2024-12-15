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

# Find intersection between lines
# Ref: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
def find_intersections(lines):
    points_on_line = []

    for i in range(0, len(lines)):
        rho = lines[i][0]
        theta = lines[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        points_on_line.append([int(x0 + 1280*(-b)), int(y0 + 1280*(a)), int(x0 - 1280*(-b)), int(y0 - 1280*(a))])
    
    points_on_line = np.array(points_on_line)
    print(points_on_line)

    #TODO: We don't need to calculate all intersections, it's enough to calculate the intersections between the vertical and horizontal lines
    intertections = []
    for i in range(0, len(points_on_line)):
        for j in range(0, len(points_on_line)):
            if i == j:
                continue
            p1 = points_on_line[i]
            p2 = points_on_line[j]
            x1 = p1[0]
            y1 = p1[1]
            x2 = p1[2]
            y2 = p1[3]
            x3 = p2[0]
            y3 = p2[1]
            x4 = p2[2]
            y4 = p2[3]
            
            x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2)*(x3 * y4 - y3*x4))/((x1 - x2) * (y3 - y4) - (y1 - y2)*(x3 - x4))
            y = ((x1 *y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4))/((x1 - x2) * (y3 - y4) - (y1 - y2)*(x3 - x4))

            if x < 0 or x > IMG_WIDTH or y < 0 or y > IMG_HEIGHT:
                continue

            if x == 0 and y == 0:
                continue
            if x == np.inf or y == np.inf:
                continue
            intertections.append([x, y])

    return np.array(intertections)


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

    #TODO: Clustering only works well if there are only lines of the paper detected. Somehow outliers need to be removed
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(mapped_lines)

    lines = kmeans.cluster_centers_
    if DEBUG:
        printed_lines = print_hough_lines(canny_img.shape, lines)
        show_and_save_image_debug(3, printed_lines)

    intersections = find_intersections(lines)

    kmeans.fit(intersections)
    intersections = np.array(kmeans.cluster_centers_, dtype="uint32")

    if DEBUG:
        plt.imshow(img, cmap="gray")
        plt.scatter(intersections[:, 0], intersections[:, 1])
        plt.savefig("debug/step4.png")
        plt.show()
    return intersections


def transform_homography(img, corners):
    #order corners
    corners = corners[corners[:,0].argsort()]
    left = np.array([corners[0], corners[1]])
    right = np.array([corners[2], corners[3]])
    left = left[left[:,1].argsort()]
    right = right[right[:,1].argsort()]
    corners = np.array([left[0], left[1], right[0], right[1]], dtype="float32")

    print(corners)

    transform_mat = cv.getPerspectiveTransform(corners, np.array([[0,0], [0, IMG_HEIGHT -1], [IMG_WIDTH - 1, 0], [IMG_WIDTH - 1, IMG_HEIGHT -1]], dtype="float32"))
    dst = cv.warpPerspective(img, transform_mat, (IMG_WIDTH, IMG_HEIGHT), flags=cv.INTER_LINEAR)
    
    if DEBUG:
        show_and_save_image_debug(5, dst)


img = plt.imread("static.png") ##Todo replace with live image
img = np.array(cv.cvtColor(img, cv.COLOR_BGR2GRAY) * 255, dtype="uint8")

corners = detect_corners(img)

img_transformed = transform_homography(img, corners)

plt.imshow(img, cmap="gray")
plt.show()