import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
import math
import time

DEBUG = True
SAVE_OUTPUT = False
IMG_HEIGHT = 720
IMG_WIDTH = 1280
TRANSFOMRED_CORNERS = np.array([[0,0], [0, IMG_HEIGHT -1], [1018, 0], [1018, IMG_HEIGHT -1]], dtype="float32")

# Maps the coins surface area to their face value
# Taken from https://www.snb.ch/de/the-snb/mandates-goals/cash/coins#t00
COINS = np.array([
        [776.83929, 5],
        [589.64553, 2],
        [422.73271, 1],
        [260.15529, 0.5],
        [348.01189, 0.2],
        [288.02318, 0.1],
        [231.00327, 0.05]])

debug_step = 1

def main():
    print("Capture start")
    #Disable debugging in live mode
    global DEBUG
    DEBUG = False
    #cv.namedWindow("cam", cv.WINDOW_NORMAL)
    cap = cv.VideoCapture(1, cv.CAP_DSHOW)
    ret = cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    success, img = cap.read()
    while True:
        sucess, img = cap.read()
        img = process_image(img)
        cv.imshow("Webcam", img)

        myKey = cv.waitKey(1)
        if myKey & 0xFF == ord('q'):  # quit when 'q' is pressed
            cap.release()
            break

        if myKey & 0xFF == ord('c'):  # capture when 'q' is pressed
            _, img = cap.read()
            capture(img)
    
    cv.destroyAllWindows()

def capture(img):
    global SAVE_OUTPUT
    SAVE_OUTPUT = True
    img = process_image(img)
    cv.imwrite("debug/capture.png", img)
    SAVE_OUTPUT = False

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
    
    points_on_line = np.array(points_on_line, dtype="float32")

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

    lines = cv.HoughLines(canny_img, 1, np.pi / 180, 200)

    if lines is None or len(lines) < 4:
        return []

    mapped_lines = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        mapped_lines.append([rho, theta])
    
    lines = np.array(mapped_lines)
    
    printed_lines = print_hough_lines(canny_img.shape, lines)
    log_step(printed_lines)

    #TODO: Clustering only works well if there are only lines of the paper detected. Somehow outliers need to be removed
    _, _, lines = cv.kmeans(lines, 4, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv.KMEANS_RANDOM_CENTERS)

    printed_lines = print_hough_lines(canny_img.shape, lines)
    log_step(printed_lines)

    intersections = find_intersections(lines)

    if len(intersections) < 4:
        return []

    _, _, intersections = cv.kmeans(intersections, 4, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv.KMEANS_RANDOM_CENTERS)
    intersections = np.array(intersections, dtype="uint32")

    #TODO: handle debug and save output properly
    if DEBUG:
        plt.imshow(img, cmap="gray")
        plt.scatter(intersections[:, 0], intersections[:, 1])
        plt.savefig("debug/step4.png")
        plt.show()
    
    #order corners
    intersections = intersections[intersections[:,0].argsort()]
    left = np.array([intersections[0], intersections[1]])
    right = np.array([intersections[2], intersections[3]])
    left = left[left[:,1].argsort()]
    right = right[right[:,1].argsort()]
    return np.array([left[0], left[1], right[0], right[1]], dtype="float32")



def transform_homography_image(img, corners):
    transform_mat = cv.getPerspectiveTransform(corners, TRANSFOMRED_CORNERS)
    dst = cv.warpPerspective(img, transform_mat, (1018, IMG_HEIGHT), flags=cv.INTER_LINEAR)
    
    log_step(dst, cmap="viridis")
    return dst

def transform_homography_mask(mask, corners):
    transform_mat = cv.getPerspectiveTransform(TRANSFOMRED_CORNERS, corners)
    dst = cv.warpPerspective(mask, transform_mat, (IMG_WIDTH, IMG_HEIGHT), flags=cv.INTER_LINEAR)
    
    log_step(dst, cmap="viridis")
    return dst


#Returns a mask for the outline of the coins aswell as the centers of each coin
def find_coins(img):
    #Threshold the image
    gray = np.array(cv.cvtColor(img,cv.COLOR_BGR2GRAY) * 255, dtype="uint8")
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    log_step(thresh)

    #Fill holes
    kernel =  np.ones((3,3), dtype="uint8")
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 3)

    log_step(opening)
    
    #Get the starting points for the watershed
    sure_bg = cv.dilate(thresh,kernel, iterations=3)
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)

    log_step(dist_transform)

    ret, sure_fg = cv.threshold(dist_transform,0.3*dist_transform.max(),255,0)
    sure_fg = np.array(sure_fg, dtype="uint8")
    unknown = cv.subtract(sure_bg, sure_fg)

    log_step(sure_fg)
    
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    markers[markers == 1] = 0

    log_step(markers, cmap="prism")

    #Create the mask
    mask = np.zeros((IMG_HEIGHT, 1018))
    mask[markers == -1] = 255

    log_step(mask)

    markers = markers - 1
    markers[markers <= 0] = 0
    areas = np.unique(markers, return_counts=True)
    
    return mask, np.array(areas)

def count_coins(coins):
    # Calculate the approximative area of the detected coins
    sum = 0
    for i in range(1, len(coins[0])): #Label 0 is background/border so start at index 1
        area = coins[1][i] * 0.0850713889 # Size of A4: 210 x 297 mm -> 1 pixel = 210 / 720 = 0.29167 mm x 0.29167 mm => 0.0850713889 mm2
        #remove outliers
        if area > max(COINS[:,0]) * 1.1 or area < min(COINS[:,0]) * 0.9:
            continue
        idx = (np.abs(COINS[:,0] - area)).argmin()
        sum += COINS[idx, 1]
    
    return sum

def process_image(img):
    img_gray = np.array(cv.cvtColor(img, cv.COLOR_BGR2GRAY) * 255, dtype="uint8")

    corners = detect_corners(img_gray)

    if len(corners) == 0:
        return img

    img_transformed = transform_homography_image(img, corners)
    mask, areas = find_coins(img_transformed)
    value = count_coins(areas)

    mask = transform_homography_mask(mask, corners)
    mask = cv.dilate(mask, np.ones((3,3), dtype="uint8"), iterations=1)

    highlighted_img = img
    highlighted_img[mask != 0] = [255, 0, 0]

    highlighted_img = cv.putText(highlighted_img, f"{value} CHF", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv.LINE_AA)

    global debug_step
    debug_step = 1
    return highlighted_img


main()

# uncomment for static test image
# img = cv.imread("static.png") ##Todo replace with live image

# highlighted_img = process_image(img)

# plt.imshow(highlighted_img)
# plt.show()

