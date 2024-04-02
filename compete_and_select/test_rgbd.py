import matplotlib.pyplot as plt
import pyk4a

from camera import Camera


def main():    
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

    # Get depth point cloud
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

    camera.close()

if __name__ == '__main__':
    main()

# image = Image.open("irl_capture.png")
# generate_action_candidates(image, "put a block in the drawer")
