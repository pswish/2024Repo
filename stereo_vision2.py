import time
import numpy as np
import cv2
from pykinect import nui

class KinectCamera:
    def __init__(self, index):
        self.index = index
        self.kinect = nui.Runtime()
        self.color_frame = None
        self.depth_frame = None

        # Set up the color frame event handler
        self.kinect.video.frame_ready += self.color_frame_ready

    def color_frame_ready(self, frame):
        """Callback function for color frame"""
        self.color_frame = frame.image.copy()

    def get_color_frame(self):
        """Get the latest color frame"""
        return self.color_frame

    def close(self):
        """Close the Kinect device"""
        self.kinect.close()

def main():
    try:
        # Initialize two Kinect cameras
        kinect1 = KinectCamera(0)
        kinect2 = KinectCamera(1)

        print("Press 'q' to quit")
        print("Press 's' to save images")

        while True:
            frame1 = kinect1.get_color_frame()
            frame2 = kinect2.get_color_frame()

            if frame1 is not None:
                cv2.imshow('Kinect 1', frame1)

            if frame2 is not None:
                cv2.imshow('Kinect 2', frame2)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                if frame1 is not None:
                    cv2.imwrite('camera1_{}.jpg'.format(timestamp), frame1)
                if frame2 is not None:
                    cv2.imwrite('camera2_{}.jpg'.format(timestamp), frame2)
                print("Images saved with timestamp: {}".format(timestamp))

    except Exception as e:
        print("Error: {}".format(str(e)))

    finally:
        kinect1.close()
        kinect2.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
