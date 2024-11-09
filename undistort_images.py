import cv2 
import numpy as np 
import glob 

def undistort_image(img, mtx, dist): 
    h, w = img.shape[:2] 
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h)) 
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx) 
    x, y, w, h = roi 
    dst = dst[y:y+h, x:x+w] 
    return dst 

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist): 
    mean_error = 0 
    for i in range(len(objpoints)): 
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist) 
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2) 
        mean_error += error 
    return mean_error/len(objpoints) 

# Load calibration data 
with np.load('calibration_data.npz') as X: 
    mtx, dist, rvecs, tvecs = X['mtx'], X['dist'], X['rvecs'], X['tvecs'] 

# Prepare object points
chessboard_size = (5, 8)  # Adjust this to match your chessboard size
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Load and process images 
images = glob.glob('/home/pswish/Downloads/photos/calibration_images/*.JPEG') 

for fname in images: 
    img = cv2.imread(fname) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)
    
    undistorted_img = undistort_image(img, mtx, dist)   
    
    # Resize for display
    img_resized = cv2.resize(img, (1024, 768))
    undistorted_img_resized = cv2.resize(undistorted_img, (1024, 768))

    # Display original and undistorted images side by side 
    combined = np.hstack((img_resized, undistorted_img_resized)) 
    cv2.imshow('Original vs Undistorted', combined) 
    cv2.waitKey(0) 

cv2.destroyAllWindows()

# Calculate reprojection error 
error = calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist) 
print(f"Mean reprojection error: {error}") 

# Experiment with different numbers of calibration images
for num_images in [5, 10, 15, 20, len(images)]:
    subset_objpoints = objpoints[:num_images]
    subset_imgpoints = imgpoints[:num_images]
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(subset_objpoints, subset_imgpoints, gray.shape[::-1], None, None)
    
    error = calculate_reprojection_error(subset_objpoints, subset_imgpoints, rvecs, tvecs, mtx, dist)
    print(f"Mean reprojection error with {num_images} images: {error}")
