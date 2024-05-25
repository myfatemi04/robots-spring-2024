### Combines the object segmentation and grasp generation steps.

import pickle
import time

import cv2
import PIL.Image as Image
from matplotlib import pyplot as plt

from .. import segment_point_cloud
from ..grasp_heuristic.detect_grasps import detect_grasps
from ..object_detection.detect_objects import detect as detect_objects_2d
from ..util.set_axes_equal import set_axes_equal


def smoothen_pcd(pcd):
    pcd_smooth = cv2.GaussianBlur(pcd, (5, 5), 2) # type: ignore
    pcd_smooth[pcd == -10000] = -10000
    return pcd_smooth

def test():
    capture_num = 1
    with open(f"capture_{capture_num}.pkl", "rb") as f:
        (rgbs, pcds) = pickle.load(f)

    # adjust point clouds
    # pcds = [smoothen_pcd(pcd) for pcd in pcds]
    mask0 = pcds[0] == -10000
    mask1 = pcds[1] == -10000
    pcds[0][..., 0] += 0.05
    pcds[0][..., 2] += 0.015
    pcds[1][..., 2] += 0.015

    pcds[0][mask0] = -10000
    pcds[1][mask1] = -10000

    # Render test
    render_point_cloud = False
    if render_point_cloud:
        render_pcd = pcds[0]
        mask = ~(render_pcd == -10000).any(axis=-1)
        render_pcd = render_pcd[mask]
        color = rgbs[0][mask]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title("Base Point Cloud Detection")
        ax.scatter(render_pcd[:, 0], render_pcd[:, 1], render_pcd[:, 2], c=color/255.0, s=0.5)
        set_axes_equal(ax)
        plt.show()

    # detect a cup
    imgs = [Image.fromarray(rgb) for rgb in rgbs]

    start_time = time.time()
    cup_detections = detect_objects_2d(imgs[0], 'cup')
    end_time = time.time()
    print("### Made 2D Detections with OwlV2 ###")
    print("Time:", end_time - start_time)
    print("Hz:", 1/(end_time - start_time))

    # generate object segmentations
    pcd_segmenter = segment_point_cloud.SamPointCloudSegmenter(device='cuda', render_2d_results=False)

    fig = plt.figure()
    for i, detection in enumerate(cup_detections):
        box = detection['box']
        x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']

        start_time = time.time()
        segmented_pcd_xyz, segmented_pcd_color, _segmentation_masks = pcd_segmenter.segment(imgs[0], pcds[0], [x1, y1, x2, y2], [imgs[1]], [pcds[1]])
        end_time = time.time()

        print(f"### Made 3D Object Segmentations with SAM [{i + 1}/{len(cup_detections)}] ###")
        print("Time:", end_time - start_time)
        print("Hz:", 1/(end_time - start_time))

        # render the cup detection
        ax = fig.add_subplot(2, len(cup_detections), 1 + i)
        ax.set_title("2D Segmentation")
        ax.imshow(imgs[0])
        ax.imshow(_segmentation_masks[0], alpha=0.5)
        ax.axis('off')

        # render the segmented point cloud
        ax = fig.add_subplot(2, len(cup_detections), len(cup_detections) + 1 + i, projection='3d')
        ax.set_title(f"3D Segmentation {i+1}/{len(cup_detections)}")
        ax.scatter(segmented_pcd_xyz[:, 0], segmented_pcd_xyz[:, 1], segmented_pcd_xyz[:, 2], c=segmented_pcd_color/255.0, s=0.5)

        start_time = time.time()

        grasps = detect_grasps(
            segmented_pcd_xyz,
            segmented_pcd_color,
            voxel_size=0.005,
            min_points_in_voxel=2,
            gripper_width=0.2,
            max_alpha=15,
            hop_size=1,
            window_size=2,
            top_k_per_angle=5,
            show_rotated_voxel_grids=False
        )

        end_time = time.time()

        print("### Generated Grasps using Force Closure ###")
        print("Time:", end_time - start_time)
        print("Hz:", 1/(end_time - start_time))

        # detect grasps
        # plot grasp locations
        for grasp in grasps:
            _worst_alpha, start, end = grasp
            (x1, y1, z1) = (start)# - lower_bound) / voxel_size
            (x2, y2, z2) = (end)# - lower_bound) / voxel_size
            ax.scatter(x1, y1, z1, c='r')
            ax.scatter(x2, y2, z2, c='g')
            ax.plot([x1, x2], [y1, y2], [z2, z2], c='b')

        set_axes_equal(ax)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test()

"""
### Made 2D Detections with OwlV2 ###
Time: 0.7501897811889648
Hz: 1.3329960299047456
No points found in the bounding box.
### Made 3D Object Segmentations with SAM [1/4] ###
Time: 0.21892690658569336
Hz: 4.567734572217031
### Generated Grasps using Force Closure ###
Time: 0.12975716590881348
Hz: 7.706703464090357
Segmenting based on bounding box [403.0, 437.0, 453.0, 528.0]
### Made 3D Object Segmentations with SAM [2/4] ###
Time: 0.40863943099975586
Hz: 2.4471451459137272
### Generated Grasps using Force Closure ###
Time: 0.1220245361328125
Hz: 8.195073152432162
Segmenting based on bounding box [637.0, 323.0, 667.0, 362.0]
### Made 3D Object Segmentations with SAM [3/4] ###
Time: 0.40489697456359863
Hz: 2.4697640704226265
### Generated Grasps using Force Closure ###
Time: 0.16212677955627441
Hz: 6.168012482187736
Segmenting based on bounding box [417.0, 325.0, 453.0, 392.0]
### Made 3D Object Segmentations with SAM [4/4] ###
Time: 0.40045857429504395
Hz: 2.4971371926805963
### Generated Grasps using Force Closure ###
Time: 0.23848891258239746
Hz: 4.193067045221659
"""
