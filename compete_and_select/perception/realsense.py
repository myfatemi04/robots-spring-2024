import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt

def init_pipeline():
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align, clipping_distance

class RealsenseWrapper:
    def __init__(self):
        pipeline, align, clipping_distance = init_pipeline()
        self.pipeline = pipeline
        self.align = align
        self.clipping_distance = clipping_distance
        
    def __iter__(self):
        return self
    
    def __next__(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        return (color_frame, aligned_depth_frame)

    def __del__(self):
        self.pipeline.stop()

def show_realsense_stream():
    cam = RealsenseWrapper()
    for color_frame, aligned_depth_frame in cam:
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # blur the depth image
        depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        bg_removed = np.zeros((480, 640, 3), dtype=np.uint8) + grey_color
        mask = (depth_image > cam.clipping_distance) | (depth_image <= 0)
        bg_removed[mask] = color_image[mask]
        
        plt.subplot(1, 2, 1)
        plt.imshow(bg_removed, cmap='jet', vmin=0, vmax=1000)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(color_image)
        plt.axis('off')
        plt.pause(0.001)

if __name__ == '__main__':
    show_realsense_stream()
