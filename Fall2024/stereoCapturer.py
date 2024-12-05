import cv2
import time

left_cam = cv2.VideoCapture(0)
right_cam = cv2.VideoCapture(2)

count = 0
last_capture_time = 0

while True:
    ret_left, frame_left = left_cam.read()
    ret_right, frame_right = right_cam.read()

    if not ret_left or not ret_right:
        print("Failed to capture frames")
        break

    cv2.imshow('Left Camera', frame_left)
    cv2.imshow('Right Camera', frame_right)

    key = cv2.waitKey(1) & 0xFF

    current_time = time.time()
    if key == ord('c') and current_time - last_capture_time > 1:
        cv2.imwrite('left_{0}.png'.format(count), frame_left)
        cv2.imwrite('right_{0}.png'.format(count), frame_right)
        print("Saved image pair {0}".format(count))
        count += 1
        last_capture_time = current_time

    elif key == ord('q'):
        break

left_cam.release()
right_cam.release()
cv2.destroyAllWindows()
