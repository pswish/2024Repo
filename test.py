import cv2
import numpy as np
import glob

def process_image(fname):
    img = cv2.imread(fname)
    img = cv2.resize(img, (1024, 768))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    ret, corners = cv2.findChessboardCornersSB(gray, (5,8), None)
    
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(img, (5,8), corners2, ret)
        print(f"Found corners in {fname}")
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
    else:
        print(f"Failed to find corners in {fname}")
    
    return ret, corners if ret else None

images = glob.glob('/home/pswish/Downloads/photos/calibration_images/*.JPEG')
print(f"Number of images found: {len(images)}")

successful_images = 0
for fname in images:
    ret, _ = process_image(fname)
    if ret:
        successful_images += 1

print(f"Successfully processed {successful_images} out of {len(images)} images")

cv2.destroyAllWindows()
