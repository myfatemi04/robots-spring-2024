# from profile import profile
from typing import List

import segment_point_cloud
from detect_grasps import detect_grasps
from generate_object_candidates import detect as detect_objects_2d

pcd_segmenter = segment_point_cloud.SamPointCloudSegmenter(device='cuda', render_2d_results=False)

class Object:
    def __init__(self, point_cloud, colors, clip_features):
        # may just store objects in world frame, since during any voxelization steps,
        # they get normalized anyway
        # self.translation = translation
        self.point_cloud = point_cloud
        self.colors = colors
        self.clip_features = clip_features

    # @profile("object.generate_grasps")
    def generate_grasps(self):
        # will need to add some kind of collision detection here next
        grasps = detect_grasps(
            self.point_cloud,
            self.colors,
            voxel_size=0.005,
            min_points_in_voxel=2,
            gripper_width=0.2,
            max_alpha=15,
            hop_size=1,
            window_size=2,
            top_k_per_angle=5,
            show_rotated_voxel_grids=False
        )
        return grasps

class Scene:
    def __init__(self, imgs, pcds, view_names):
        self.imgs = imgs
        self.pcds = pcds
        self.view_names = view_names
        
    # @profile("scene.locate")
    def locate(self, object_name, target=0):
        detections_2d = detect_objects_2d(self.imgs[target], object_name)

        base_rgb_image = self.imgs[target]
        base_point_cloud = self.pcds[target]
        supplementary_rgb_images = [self.imgs[i] for i in range(2) if i != target]
        supplementary_point_clouds = [self.pcds[i] for i in range(2) if i != target]

        objects: List[Object] = []

        for i, detection in enumerate(detections_2d):
            box = detection['box']
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            segmented_pcd_xyz, segmented_pcd_color, _segmentation_masks = \
                pcd_segmenter.segment(base_rgb_image, base_point_cloud, [x1, y1, x2, y2], supplementary_rgb_images, supplementary_point_clouds)
            
            objects.append(Object(segmented_pcd_xyz, segmented_pcd_color, clip_features=None))

        return objects

