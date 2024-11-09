import cv2
import numpy as np
import glob

def process_image(fname, objp):
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
        return True, corners2
    else:
        print(f"Failed to find corners in {fname}")
        return False, None

# Prepare object points
objp = np.zeros((5*8, 3), np.float32)
objp[:,:2] = np.mgrid[0:5, 0:8].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

images = glob.glob('/home/pswish/Downloads/photos/calibration_images/*.JPEG')
print(f"Number of images found: {len(images)}")

for fname in images:
    ret, corners = process_image(fname, objp)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

cv2.destroyAllWindows()

print(f"Successfully processed {len(objpoints)} out of {len(images)} images")

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1024, 768), None, None)

# Save the calibration results
np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

print("Calibration complete. Data saved to 'calibration_data.npz'")

# Validate on a test image (use the first image as an example)
test_img = cv2.imread(images[0])
h, w = test_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Display the original and undistorted images
cv2.imshow('Original', test_img)
cv2.imshow('Undistorted', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
