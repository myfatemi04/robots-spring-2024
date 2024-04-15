### Combines the object segmentation and grasp generation steps.

import pickle

import cv2
import PIL.Image as Image
import segment_point_cloud
from generate_object_candidates import detect as detect_objects_2d
from matplotlib import pyplot as plt
from set_axes_equal import set_axes_equal


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
    cup_detections = detect_objects_2d(imgs[0], 'cup')

    # generate object segmentations
    pcd_segmenter = segment_point_cloud.SamPointCloudSegmenter(device='cuda', render_2d_results=False)

    fig = plt.figure()
    for i, detection in enumerate(cup_detections):
        box = detection['box']
        x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        segmented_pcd_xyz, segmented_pcd_color, _segmentation_masks = pcd_segmenter.segment(imgs[0], pcds[0], [x1, y1, x2, y2], [imgs[1]], [pcds[1]])

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
        set_axes_equal(ax)


    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test()
