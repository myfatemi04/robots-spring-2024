from functools import cached_property

from .detect_grasps import detect_grasps


class Object:
    def __init__(self, point_cloud, colors, segmentation_masks, clip_features=None):
        # may just store objects in world frame to avoid having to keep track
        # of rotations and such
        self.point_cloud = point_cloud
        self.colors = colors
        self.segmentation_masks = segmentation_masks
        self.clip_features = clip_features

    @cached_property
    def centroid(self):
        return self.point_cloud.mean(axis=0)

    # at some point expose this to the model so it can generate grasps
    # and then rank them later
    def generate_grasps(self, show_rotated_voxel_grids=False):
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
            show_rotated_voxel_grids=show_rotated_voxel_grids
        )
        return grasps
