import cv2
import numpy as np
colors = [0] * 4

#detects the home color and returns it
def countPixels(mask):
	return cv2.countNonZero(mask)

image = cv2.imread("420.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 100, 100], dtype = "uint8")
upper_red = np.array([10, 255, 255], dtype = "uint8")

lower_blue = np.array([90,50,50], dtype = "uint8")
upper_blue = np.array([130,255,255], dtype = "uint8")
# b g r

lower_green = np.array([35,0,0], dtype = "uint8")
upper_green = np.array([85,255,255], dtype = "uint8")

upper_white = np.array([131,255,255], dtype=np.uint8)
lower_white = np.array([0,0,190], dtype=np.uint8)

lower_yellow = np.array([20,100,175], dtype = "uint8")
upper_yellow = np.array([91,255,255], dtype = "uint8")

lower_orange = np.array([10,100,20], dtype=np.uint8)
upper_orange = np.array([25,255,255], dtype=np.uint8)



#mask = cv2.inRange(hsv, lower_red, upper_red) | cv2.inRange(hsv, lower_blue, upper_blue) | cv2.inRange(hsv, lower_green, upper_green) | cv2.inRange(hsv, lower_white, upper_white)
#while camera is on
pixel = 0
maskRed = cv2.inRange(hsv, lower_red, upper_red)
maskBlue = cv2.inRange(hsv, lower_blue, upper_blue)
maskGreen = cv2.inRange(hsv, lower_green, upper_green)
maskWhite = cv2.inRange(hsv, lower_white, upper_white)-cv2.inRange(hsv, lower_orange, upper_orange)
maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)-cv2.inRange(hsv, lower_orange, upper_orange)-cv2.inRange(hsv, lower_green, upper_green)
maskOrange = cv2.inRange(hsv, lower_orange, upper_orange)
mask = maskRed + maskBlue + maskGreen + maskWhite + maskYellow - maskOrange #add all masks subtract orange

maxcolor = 0
homeMask = None
maxColorName = "none"
if countPixels(maskRed) > maxcolor:
	homeMask = maskRed
	maxcolor = countPixels(maskRed)
	maxColorName = "Red"
	print maxcolor
if countPixels(maskBlue) > maxcolor:
	homeMask = maskBlue
	maxcolor = countPixels(maskBlue)
	maxColorName = "Blue"
	print maxcolor
if countPixels(maskGreen) > maxcolor:
	homeMask = maskGreen
	maxcolor = countPixels(maskGreen)
	maxColorName = "Green"
	print maxcolor
if countPixels(maskYellow) > maxcolor:
	homeMask = maskYellow
	maxcolor = countPixels(maskYellow)
	maxColorName = "Yellow"
	print maxcolor

finalMask = homeMask

cv2.imshow("Original", image)
cv2.imshow("HSV Combined", maskYellow)
print(maxColorName)

cv2.waitKey()
cv2.destroyAllWindows()

