import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import polymetis
import torch
from generate_action_candidates import generate_action_candidates
from generate_object_candidates import draw_set_of_marks
from select_next_action import select_next_action

# import cv2
import apriltag


def main():
    import pyk4a
    sys.path.insert(0, "../")
    from camera import Camera
    
    num_cameras = 2
    if pyk4a.connected_device_count() < num_cameras:
        print(f"Error: Not enough K4A devices connected (<{num_cameras}).")
        exit(1)

    k4a_devices = [pyk4a.PyK4A(device_id=i) for i in range(num_cameras)]
    k4a_device_map = {}
    for device in k4a_devices:
        device.start()
        k4a_device_map[device.serial] = device

    print(k4a_device_map)

    left = Camera(k4a_device_map['000259521012'])
    right = Camera(k4a_device_map['000243521012'])
    # camera = Camera(k4a_devices[0])

    draw_all_marks = False
    do_control = True
    if do_control:
        polymetis_server_ip = "192.168.1.222"
        robot = polymetis.RobotInterface(
            ip_address=polymetis_server_ip,
            port=50051,
            enforce_version=False,
        )
        ROBOT_CONTROL_X_BIAS = 0.14
        ROBOT_CONTROL_Y_BIAS = 0.03
        ROBOT_CONTROL_Z_BIAS = 0.10
        movement_bias = torch.tensor([ROBOT_CONTROL_X_BIAS, ROBOT_CONTROL_Y_BIAS, ROBOT_CONTROL_Z_BIAS]).float()

        def move(x, y, z):
            robot.move_to_ee_pose(torch.tensor([x, y, z]).cpu().float() + movement_bias)

        # move(0.44, -0.14, 0.1)
        move(0.5, 0.0, 0.5)

        time.sleep(2)
    else:
        robot = None

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

            # only take points within a certain distance
            # point_mask = pcd_xyz[:, -1] < 3000
            # pcd_xyz = pcd_xyz[point_mask]
            # pcd_color = pcd_color[point_mask]
            skip_every = 50
            s = 0.5
            pcd_xyz = pcd_xyz[::skip_every]
            pcd_color = pcd_color[::skip_every]

            ax = plt.figure().add_subplot(projection='3d')
            plt.title("Calibrated depth cloud")
            ax.scatter(*pcd_xyz.T, c=pcd_color/255, s=s, alpha=1.0)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.show()

        instructions = input("Instruction: ")
        if len(instructions) == 0:
            break
        start_t = time.time()
        (reasoning, plan_str, plan) = generate_action_candidates(image_left, instructions)
        end_t = time.time()
        print(f"Plan generation time: {end_t-start_t:.4f}")

        act, obj = plan[0]

        start_t = time.time()
        successful, info = select_next_action(image_left, instructions, reasoning, plan_str, obj)
        end_t = time.time()
        print(f"Action selection time: {end_t-start_t:.4f}")

        if successful:
            action_selection = info['action_selection']
            detections = info['detections']

            print("::: Action Selection :::")
            print("reasoning:", action_selection['reasoning'])
            print("object id:", action_selection['object_id'])
            print("x pos:", action_selection['relative_x_percent'])
            print("y pos:", action_selection['relative_y_percent'])
            print("detections:", detections)

            detection_index = int(action_selection['object_id'].replace("#", "")) - 1

            if draw_all_marks:
                draw_set_of_marks(
                    image_left,
                    detections,
                    ['Selected Object' if i == detection_index else f"#{i+1}" for i in range(len(detections))],
                    live=True
                )
            else:
                draw_set_of_marks(
                    image_left,
                    [detections[detection_index]],
                    ['Target object'],
                    live=True
                )
            x_rel = action_selection['relative_x_percent'] / 100
            y_rel = action_selection['relative_y_percent'] / 100

            detection = detections[detection_index]
            box = detection['box']
            box = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
            w = box[2] - box[0]
            h = box[3] - box[1]
            x = x_rel * w + box[0]
            y = (1 - y_rel) * h + box[1] # y values go top to bottom

            # draw a marker in neon green over the target location
            plt.scatter(x, y, marker='x', color=(0, 1, 0), label='Precise Point')
            plt.legend()
            plt.show()

            # Now, given this 2D position, translate to robot frame.
            # Note that the depth camera can be a little spotty sometimes!
            # We should prefer to have a purely visual method instead.
            # For example, we could provide two views, and have the VLM
            # use the view information (and some axes drawn onto the images)
            # to select a final location.
            transformed_depth_pcd = cap_left.transformed_depth_point_cloud
            print(transformed_depth_pcd.shape)
            # mm to meters
            point_from_camera_perspective = transformed_depth_pcd[int(y), int(x)] / 1000
            # unsqueeze to batch dimension
            point_from_camera_perspective = point_from_camera_perspective[None, :]

            print(point_from_camera_perspective)

            # Apply AprilTag calibration to convert this 3D point to robot frame
            camera_translation = left.extrinsic_matrix[[0, 1, 2], 3]
            camera_rotation = left.extrinsic_matrix[:3, :3]

            # we change units from mm to meters
            # untranslate and unrotate
            point_robot_frame = (camera_rotation.T @ (point_from_camera_perspective - camera_translation).T).T
            point_robot_frame = point_robot_frame[0]
            
            print(point_robot_frame)
            if 'y' == input("OK to move here? (y/n) "):
                move(point_robot_frame[0], point_robot_frame[1], point_robot_frame[2])

    left.close()
    right.close()

if __name__ == '__main__':
    main()

# image = Image.open("irl_capture.png")
# generate_action_candidates(image, "put a block in the drawer")
