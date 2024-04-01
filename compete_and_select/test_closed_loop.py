import sys

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import generate_plan

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

    image = Image.fromarray(image_np)

    # generate_plan(image, "put a block in the drawer")

    plt.title("Capture")
    plt.imshow(image)
    plt.show()

    camera.close()

# if __name__ == '__main__':
#     main()

image = Image.open("irl_capture.png")
generate_plan(image, "put a block in the drawer")
