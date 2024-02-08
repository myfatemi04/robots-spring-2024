import pyk4a
import cv2
import numpy as np
import apriltag

# PyK4A requires libk4a1.4 and libk4a1.4-dev.

# Load camera with the default config
# Device names:
# 000243521012 [right camera]
# 000256121012 [left camera]

print("Num. Connected Devices:", pyk4a.connected_device_count())

k4a_left = pyk4a.PyK4A(device_id=0)# 256121012)
k4a_right = pyk4a.PyK4A(device_id=1)# 243521012)
k4a_left.start()
k4a_right.start()

apriltag_detector = apriltag.apriltag("tag36h11")

try:
    while True:
        left_capture = k4a_left.get_capture()
        right_capture = k4a_right.get_capture()

        left_color = np.ascontiguousarray(left_capture.color[:, :, :3])
        left_gray = cv2.cvtColor(left_color, cv2.COLOR_RGB2GRAY)
        tags = apriltag_detector.detect(left_gray)
        for tag in tags:
            points = tag['lb-rb-rt-lt']
            cv2.drawContours(left_color, [points.astype(int)], 0, (0, 255, 0), 3)

        right_color = np.ascontiguousarray(right_capture.color[:, :, :3])
        right_gray = cv2.cvtColor(right_color, cv2.COLOR_RGB2GRAY)
        tags = apriltag_detector.detect(right_gray)
        for tag in tags:
            points = tag['lb-rb-rt-lt']
            print(points)
            cv2.drawContours(right_color, [points.astype(int)], 0, (0, 255, 0), 3)

        # Depth capture is (576, 640)

        # print(".color.shape:", left_capture.color.shape)
        # print(".depth.shape:", left_capture.depth.shape)

        # Scale depth inversely
        ld = left_capture.depth
        ld = ld/ld.max()
        rd = right_capture.depth
        rd = rd/rd.max()

        cv2.imshow('left', left_color)
        cv2.imshow('right', right_color)
        cv2.imshow('left_depth', ld)
        cv2.imshow('right_depth', rd)

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    k4a_left.close()
    k4a_right.close()
