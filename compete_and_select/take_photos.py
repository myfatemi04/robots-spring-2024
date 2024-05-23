import os

import PIL.Image

import matplotlib.pyplot as plt
from .perception.rgbd import RGBD


def main():
    rgbd = RGBD(num_cameras=1)

    group_folder = input("Group folder: ")
    out_dir = f"photos/{group_folder}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    counter = 0
    previous_label = None

    while True:
        label = input("Label (leave blank to exit): ")
        if label == "":
            break

        rgbs, pcds = rgbd.capture()

        if label != 'test':
            if label == 'retake':
                counter -= 1
                label = previous_label

            save_path = f"photos/{group_folder}/{counter:03d}_{label}.png"

            PIL.Image.fromarray(rgbs[0]).save(save_path)

            print(f"Saved photo {counter} to {save_path}")

            counter += 1
        else:
            print("Not saving 'test' photo.")

        previous_label = label

        plt.title("Captured Image")
        plt.imshow(rgbs[0])
        plt.show()


if __name__ == '__main__':
    main()
