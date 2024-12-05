import cv2
import numpy as np
import glob

# Checkerboard dimensions
CHECKERBOARD = (8, 5)  # Adjust these numbers based on your checkerboard

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpoints_left = []  # 2d points in left image plane
imgpoints_right = []  # 2d points in right image plane

# Load images
images_left = glob.glob('left_*.png')
images_right = glob.glob('right_*.png')

for img_left, img_right in zip(images_left, images_right):
    left = cv2.imread(img_left)
    right = cv2.imread(img_right)
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

    if ret_left and ret_right:
        objpoints.append(objp)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

# Calibrate each camera individually
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

# Stereo calibration
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, 
    gray_left.shape[::-1], criteria_stereo, flags)

# Print results
print("Left camera matrix:\n", mtx_left)
print("\nRight camera matrix:\n", mtx_right)
print("\nRotation matrix:\n", rot)
print("\nTranslation vector:\n", trans)
print("\nEssential matrix:\n", essentialMatrix)
print("\nFundamental matrix:\n", fundamentalMatrix)
