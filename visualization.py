import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Add background
vertices = [
    np.array([
        (0, -1, -0.1),
        (0,  1, -0.1),
        (1,  1, -0.1),
        (1, -1, -0.1),
    ]),
    np.array([
        (0, -1, -0.1),
        (0,  1, -0.1),
        (0,  1, 1),
        (0, -1, 1),
    ]),
]

class ObjectDetectionVisualizer:
    def __init__(self, live=True):
        # For plotting 3D object detections
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(projection='3d', computed_zorder=False)
        self.axes.view_init(elev=30, azim=0, roll=0)
        self.live = live

    def show(self, objects):
        self.axes.clear()
        self.axes.set_title("World Coordinates and Detections")

        self.axes.add_collection3d(Poly3DCollection(vertices, color=(0.2, 0.2, 0.2, 1.0)))
        for object in objects:
            label, (x, y, z), size, color = object
            self.axes.scatter(x, y, z, label=label, s=size, c=color)
        self.axes.set_xlabel("X (m)")
        self.axes.set_ylabel("Y (m)")
        self.axes.set_zlabel("Z (m)")
        self.axes.set_xlim(-1, 1)
        self.axes.set_ylim(-1, 1)
        self.axes.set_zlim(-1, 1)
        self.axes.legend()
        if self.live:
            plt.pause(0.1)
        else:
            plt.show()
