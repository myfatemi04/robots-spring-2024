from panda import Panda
from rotation_utils import vector2quat
import time

# Robot will hold a block in its gripper and face forward.
# Then it will move the block to a set of defined corners in robot frame.

wait = lambda: input("Waiting")

robot = Panda()

robot.move_to(
    [0.4, 0, 0.4],
    vector2quat([0, 0, -1], [0, -1, 0])
)
wait()

robot.stop_grasp()
time.sleep(5)
robot.start_grasp()

robot.move_to([0.2, 0.1, 0.1])
wait()

robot.move_to([0.2, -0.1, 0.1])
wait()

robot.move_to([0.3, -0.1, 0.1])
wait()

robot.move_to([0.3, 0.1, 0.1])
wait()

robot.stop_grasp()
