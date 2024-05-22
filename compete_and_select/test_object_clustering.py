# Create a 3D object segmentation from point clouds.
import matplotlib.pyplot as plt
import numpy as np
import hdbscan

from compete_and_select.perception.rgbd import RGBD

def create_clusters(pcd, colors):
    # Turn each point into a single vector, such that Euclidean distance resembles approx. probability
    # that points belong to the same object.
    # Colors passed as arguments should be scaled according to the weight we want to apply.
    pvecs = np.concatenate([pcd, colors], axis=-1)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=0.1, gen_min_span_tree=True)
    clusterer.fit(pvecs)
    return (clusterer.labels_, clusterer.probabilities_)

def main():
    rgbd = RGBD()

    try:
        while True:
            (rgbs, pcds) = rgbd.capture()

            print([pcd is not None for pcd in pcds])

            if not all(pcd is not None for pcd in pcds):
                continue
            
            # render point cloud clustering result
            pcd_flat = np.concatenate([pcd.reshape(-1, 3) for pcd in pcds], axis=0)
            rgb_flat = np.concatenate([rgb.reshape(-1, 3) for rgb in rgbs], axis=0)
            pcd_flat = pcd_flat[::20]
            rgb_flat = rgb_flat[::20]

            clusters, probs = create_clusters(pcd_flat, 0.1 * (rgb_flat / 255))

            # visualize according to different colors
            ax = plt.figure().add_subplot(projection='3d')
            plt.title("Clustered depth cloud")

            num_clusters = clusters.max()
            for i in range(num_clusters):
                cluster = pcd_flat[clusters == i]
                ax.scatter(*cluster.T, s=0.5, alpha=1.0)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_aspect("equal")
            plt.show()
        
    except KeyboardInterrupt:
        pass

    rgbd.close()

if __name__ == '__main__':
    main()
