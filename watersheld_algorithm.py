import cv2
import numpy as np

image = cv2.imread("coin.jpeg")
original_image = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel,iterations=2)

sure_bg = cv2.dilate(opening,kernel,iterations=3)

distance_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
_ , sure_fg = cv2.threshold(distance_transform, 0.7*distance_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg,sure_fg)

_, makers = cv2.connectedComponents(sure_fg)

makers = makers + 1

makers[unknown == 255] =0

cv2.watershed(image,makers)

image[makers == -1] = [0,0,255]


# Display the results
cv2.imshow("Original Image", original_image)
cv2.imshow("Binary Image", binary)
cv2.imshow("Segmented Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
