import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from generate_action_candidates import generate_action_candidates
from generate_object_candidates import draw_set_of_marks
from select_next_action import select_next_action

def main():
    import pyk4a
    sys.path.insert(0, "../")
    from camera import Camera
    
    num_cameras = 1
    if pyk4a.connected_device_count() < num_cameras:
        print(f"Error: Not enough K4A devices connected (<{num_cameras}).")
        exit(1)

    k4a_devices = [pyk4a.PyK4A(device_id=i) for i in range(num_cameras)]
    k4a_device_map = {}
    for device in k4a_devices:
        device.start()
        k4a_device_map[device.serial] = device

    # left = Camera(k4a_device_map['000256121012'])
    # right = Camera(k4a_device_map['000243521012'])
    camera = Camera(k4a_devices[0])
    cap = camera.capture()
    image_np = cap.color[..., :-1]
    image_np = image_np[..., ::-1]
    image_np = np.ascontiguousarray(image_np)

    # Get depth point cloud
    show_pcd = False

    if show_pcd:
        pcd_xyz = cap.depth_point_cloud.reshape(-1, 3)
        pcd_color = cap.transformed_color.reshape(-1, 4)[:, :-1][:, ::-1]

        point_mask = pcd_xyz[:, -1] < 1000
        pcd_xyz = pcd_xyz[point_mask]
        pcd_color = pcd_color[point_mask]
        skip_every = 5
        pcd_xyz = pcd_xyz[::skip_every]
        pcd_color = pcd_color[::skip_every]

        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(*pcd_xyz.T, c=pcd_color/255)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    image = Image.fromarray(image_np)
    plt.imshow(image)
    plt.show()

    instructions = input("Instruction: ")
    start_t = time.time()
    (reasoning, plan_str, plan) = generate_action_candidates(image, instructions)
    end_t = time.time()
    print(f"Plan generation time: {end_t-start_t:.4s}")

    act, obj = plan[0]

    start_t = time.time()
    successful, info = select_next_action(image, instructions, reasoning, plan_str, obj)
    end_t = time.time()
    print(f"Action selection time: {end_t-start_t:.4s}")

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

        draw_set_of_marks(
            image,
            detections,
            ['Selected Object' if i == detection_index else f"#{i+1}" for i in range(len(detections))],
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
        y = y_rel * h + box[1]

        # draw a marker in neon green over the target location
        plt.scatter(x, y, marker='x', color=(0, 1, 0), label='Precise Point')
        plt.legend()
        plt.show()

    camera.close()

if __name__ == '__main__':
    main()

# image = Image.open("irl_capture.png")
# generate_action_candidates(image, "put a block in the drawer")
