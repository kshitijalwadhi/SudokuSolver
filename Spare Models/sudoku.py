import cv2
import numpy as np

img_orig = cv2.imread('sudoku.jpg')
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(img, (5, 5), 0)

thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
c = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = i
            img = cv2.drawContours(img, contours, c, (0, 255, 0), 3)
    c += 1


mask = np.zeros((img.shape), np.uint8)
cv2.drawContours(mask, [best_cnt], 0, 255, -1)
cv2.drawContours(mask, [best_cnt], 0, 0, 2)
cv2.imshow('mask', mask)

out = np.zeros_like(img)
out[mask == 255] = img[mask == 255]
cv2.imshow('NewImg', out)

blur = cv2.GaussianBlur(out, (5, 5), 0)
cv2.imshow('blur1', blur)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
cv2.imshow('thresh1', thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

c = 0

for i in contours:
    area = cv2.contourArea(i)
    if area > 1000/2:
        cv2.drawContours(thresh, contours, c, (255, 0, 0), 3)
    c += 1

cv2.imshow('final image', thresh)


#cv2.imshow('sudoko', img)
#cv2.imshow('blur', blur)
#cv2.imshow('thresh', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
