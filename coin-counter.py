import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
from numpy import unravel_index
import math
from sklearn.cluster import KMeans

DEBUG = True
SAVE_OUTPUT = True
IMG_HEIGHT = 720
IMG_WIDTH = 1280

debug_step = 1

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

def log_step(img, cmap = "gray"):    
    if not DEBUG and not SAVE_OUTPUT:
        return
    
    global debug_step
    
    plt.imshow(img, cmap=cmap)
    if DEBUG:
        plt.show()
    if SAVE_OUTPUT:
        cv.imwrite(f"debug/step{debug_step}.png", img)
    debug_step += 1

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

    log_step(canny_img)

    lines = cv.HoughLines(canny_img, 1, np.pi / 180, 150)
    mapped_lines = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        mapped_lines.append([rho, theta])
    
    print(mapped_lines)
    lines = mapped_lines
    
    printed_lines = print_hough_lines(canny_img.shape, lines)
    log_step(printed_lines)

    #TODO: Clustering only works well if there are only lines of the paper detected. Somehow outliers need to be removed
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(mapped_lines)

    lines = kmeans.cluster_centers_

    printed_lines = print_hough_lines(canny_img.shape, lines)
    log_step(printed_lines)

    intersections = find_intersections(lines)

    kmeans.fit(intersections)
    intersections = np.array(kmeans.cluster_centers_, dtype="uint32")

    #TODO: handle debug and save output properly
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
    
    log_step(dst, cmap="viridis")
    return dst

#Returns a mask for the outline of the coins aswell as the centers of each coin
def mask_coins(img):
    #Threshold the image
    gray = np.array(cv.cvtColor(img,cv.COLOR_BGR2GRAY) * 255, dtype="uint8")
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    log_step(thresh)
    
    #Get the starting points for the watershed
    kernel =  np.ones((3,3), dtype="uint8")
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv.dilate(thresh,kernel, iterations=3)
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    sure_fg = np.array(sure_fg, dtype="uint8")
    unknown = cv.subtract(sure_bg, sure_fg)

    log_step(sure_fg)
    
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    log_step(markers, cmap="prism")

    markers = cv.watershed(img, markers)
    #Create the mask
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    mask[markers == -1] = 255

    log_step(mask)
    
    return mask




img = cv.imread("static.png") ##Todo replace with live image

plt.imshow(img)
plt.show()

img_gray = np.array(cv.cvtColor(img, cv.COLOR_BGR2GRAY) * 255, dtype="uint8")

corners = detect_corners(img_gray)

img_transformed = transform_homography(img, corners)
mask, centers = mask_coins(img_transformed)

plt.imshow(img)
plt.show()