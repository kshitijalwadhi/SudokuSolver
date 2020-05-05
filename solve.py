from __future__ import print_function
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from imutils.perspective import four_point_transform, order_points
from skimage.segmentation import clear_border

from sudokusolver import *

image = cv2.imread("sudoku3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
gray = cv2.dilate(gray, kernel)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)


thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# cv2.imshow("Thresholded", thresh)


# find contours
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# sort contours decreasing order area wise
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
mask = np.zeros((thresh.shape), np.uint8)
c = cnts[0]

clone = image.copy()

peri = cv2.arcLength(c, closed=True)
poly = cv2.approxPolyDP(c, epsilon=0.02 * peri, closed=True)

if len(poly) == 4:
    cv2.drawContours(clone, [poly], -1, (0, 0, 255), 2)

    warped = four_point_transform(image, poly.reshape(-1, 2))
    cv2.imshow("contours", clone)
    cv2.imshow("warped", warped)


warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
winX = int(warped.shape[1] / 9.0)
winY = int(warped.shape[0] / 9.0)

model = load_model("mnistModel.h5")
# model = load_model("MNIST_keras_CNN.h5")
# model = load_model("try1.model")
# model = load_model("rnn.model")

# empty lists to capture recognized digits and center co-ordinates of the cells
labels = []
centers = []

predictions = []
for y in xrange(0, warped.shape[0], winY):
    for x in xrange(0, warped.shape[1], winX):

        window = warped[y : y + winY, x : x + winX]

        if window.shape[0] != winY or window.shape[1] != winX:
            continue

        clone = warped.copy()
        digit = cv2.resize(window, (28, 28))
        # cv2.imshow("digit", digit)
        # 90 works for sudoku3.jpg
        # increasing this number allows better readability
        _, digit2 = cv2.threshold(digit, 120, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("digit2", digit2)
        digit3 = clear_border(digit2)
        cv2.imshow("digit3", digit3)
        numpixel = cv2.countNonZero(digit3)
        _, digit4 = cv2.threshold(digit3, 0, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("digit4", digit4)
        # print(numpixel)
        if numpixel < 20:
            # print("0")
            label = 0
        else:
            # label = 1
            _, digit4 = cv2.threshold(digit4, 0, 255, cv2.THRESH_BINARY_INV)
            # digit4 = tf.keras.utils.normalize(digit4, axis=1)
            digit4 = digit4 / 255.0
            # print(digit4)
            array = model.predict(digit4.reshape(1, 28, 28, 1))
            # array = model.predict(digit4.reshape(1, 28, 28))
            label = np.argmax(array)
            # label = model.predict_classes([digit3.reshape(1, 28, 28, 1)])[0]
        # print(label)
        labels.append(label)
        centers.append(((x + x + winX) // 2, (y + y + winY + 6) // 2))

        cv2.rectangle(clone, (x, y), (x + winX, y + winY), (0, 255, 0), 2)
        # cv2.imshow("Window", clone)
        # cv2.waitKey(0)


grid = np.array(labels).reshape(9, 9)

zero_indices = zip(*np.where(grid == 0))
zero_centres = np.array(centers).reshape(9, 9, 2)


print("Unsolved:")
for i in range(9):
    for j in range(9):
        temp = grid[i][j]
        if temp == 0:
            temp = "."
        print(temp, end=" ")
    print()

flag = solve_sudoku(grid)

if flag == 1:
    print("Solved:")
    printMatrix(grid)
    warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    for row, col in zero_indices:
        cv2.putText(
            warped,
            str(grid[row][col]),
            (zero_centres[row][col][0] - 10, zero_centres[row][col][1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
        )

    cv2.imshow("Solved", warped)
    cv2.waitKey(0)

    pt_src = [
        [0, 0],
        [warped.shape[1], 0],
        [warped.shape[1], warped.shape[0]],
        [0, warped.shape[0]],
    ]
    pt_src = np.array(pt_src, dtype="float")

    pt_dst = poly.reshape(4, 2)
    pt_dst = pt_dst.astype("float")

    pt_src = order_points(pt_src)
    pt_dst = order_points(pt_dst)

    # homography matrix for transformation of points from one image to other

    H, _ = cv2.findHomography(pt_src, pt_dst)

    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    im_out = cv2.warpPerspective(warped, H, dsize=(gray.shape[1], gray.shape[0]))
    # im_out = cv2.addWeighted(gray, 0.9, im_out, 0.2, 0)
    cv2.addWeighted(image, 0.5, im_out, 0.5, 0, im_out)
    # cv2.imshow("finally", final)
    cv2.imshow("Projected", im_out)
    cv2.waitKey(0)
else:
    print("Can't be solved.")


cv2.waitKey(0)
cv2.destroyAllWindows()
