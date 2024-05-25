import matplotlib.pyplot as plt
import sys
import apriltag
import numpy as np
import pickle
import PIL.Image as Image

def main():
    import pyk4a
    sys.path.insert(0, "../")
    from .camera import Camera

    num_cameras = 2
    if pyk4a.connected_device_count() < num_cameras:
        print(f"Error: Not enough K4A devices connected (<{num_cameras}).")
        exit(1)

    k4a_devices = [pyk4a.PyK4A(device_id=i) for i in range(num_cameras)]
    k4a_device_map = {}
    for device in k4a_devices:
        device.start()
        k4a_device_map[device.serial] = device

    left = Camera(k4a_device_map['000256121012'])
    right = Camera(k4a_device_map['000243521012'])
    
    apriltag_detector = apriltag.apriltag("tag36h11")
    apriltag_object_points = np.array([
        # order: left bottom, right bottom, right top, left top
        [1, -1/2, 0],
        [1, +1/2, 0],
        [0, +1/2, 0],
        [0, -1/2, 0],
    ]).astype(np.float32) * 0.1778

    apriltag_image_points = None
    for iteration in range(1):
        cap_left = left.capture()
        image_np_left = cap_left.color[..., :-1]
        image_np_left = image_np_left[..., ::-1]
        image_np_left = np.ascontiguousarray(image_np_left)
        image_left = Image.fromarray(image_np_left)

        cap_right = right.capture()
        image_np_right = cap_right.color[..., :-1]
        image_np_right = image_np_right[..., ::-1]
        image_np_right = np.ascontiguousarray(image_np_right)
        # image_right = Image.fromarray(image_np_right)

        # Ensure that extrinsic matrices are calibrated
        if left.extrinsic_matrix is None:
            left_gray = image_np_left.mean(axis=-1).astype(np.uint8)
            detections = apriltag_detector.detect(left_gray)
            if len(detections) == 1:
                detection = detections[0]
                apriltag_image_points = detection['lb-rb-rt-lt']
                left.infer_extrinsics(apriltag_image_points, apriltag_object_points)
                calib = left.export_calibration()
                with open("calib.left.pkl", "wb") as f:
                    pickle.dump(calib, f)

        if right.extrinsic_matrix is None:
            right_gray = image_np_right.mean(axis=-1).astype(np.uint8)
            detections = apriltag_detector.detect(right_gray)
            if len(detections) == 1:
                detection = detections[0]
                apriltag_image_points = detection['lb-rb-rt-lt']
                right.infer_extrinsics(apriltag_image_points, apriltag_object_points)
                calib = right.export_calibration()
                with open("calib.right.pkl", "wb") as f:
                    pickle.dump(calib, f)

        ### Render RGBD data ###
        plt.title("RGBD superimposition")
        plt.imshow(image_left)

        depth_map = cap_left.transformed_depth
        scaled_depth = depth_map.max()/(depth_map.max() + depth_map)
        scaled_depth[depth_map == 0] = 0
        plt.imshow(scaled_depth, cmap='magma', alpha=scaled_depth)

        # Draw apriltag image points, if they are found
        if apriltag_image_points is not None:
            print(apriltag_image_points)
        else:
            print("apriltag not yet detected.")

        plt.colorbar()
        plt.show()

        if left.extrinsic_matrix is not None and right.extrinsic_matrix is not None:
            # Render point cloud in robot frame (calibrated with the AprilTag)
            # Note you can also use cap.transformed_depth_point_cloud.
            pcd_xyz = np.concatenate([
                right.transform_sensed_points_to_robot_frame(cap_right.transformed_depth_point_cloud).reshape(-1, 3),
                left.transform_sensed_points_to_robot_frame(cap_left.transformed_depth_point_cloud).reshape(-1, 3),
            ], axis=0)
            pcd_color = np.concatenate([
                cap_right.color.reshape(-1, 4)[:, :-1][:, ::-1],
                cap_left.color.reshape(-1, 4)[:, :-1][:, ::-1],
            ], axis=0)

            # filter out points outside of the workspace.
            # this means points behind the robot or more than 1 meter away.
            # (just for viewing purposes)
            mask = (pcd_xyz[:, 0] > -0.5) & (np.linalg.norm(pcd_xyz, axis=-1) < 1)
            pcd_xyz = pcd_xyz[mask]
            pcd_color = pcd_color[mask]

            # only take points within a certain distance
            # point_mask = pcd_xyz[:, -1] < 3000
            # pcd_xyz = pcd_xyz[point_mask]
            # pcd_color = pcd_color[point_mask]
            skip_every = 10
            s = 1.5
            pcd_xyz = pcd_xyz[::skip_every]
            pcd_color = pcd_color[::skip_every]

            ax = plt.figure().add_subplot(projection='3d')
            plt.title("Calibrated depth cloud")
            ax.scatter(*pcd_xyz.T, c=pcd_color/255, s=s, alpha=1.0)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_aspect("equal")
            plt.show()


if __name__ == '__main__':
    main()
