# Instructions

You are writing code to control a (hypothetical) robot. Assume you have access to the following APIs. If you require clarification about anything
whatsoever, you shall call the `ask()` function.

## API Documentation

### `Object` class

This is a class used to represent objects in a scene.

Attributes:
 - point_cloud: N x 6 matrix of (x, y, z, r, g, b) points
 - centroid: (x, y, z) position

Note: Positions are specified in meters.

### `Scene` class

Methods:

1. `scene.choose(object_type, object_description)`

Description:
Helps you locate objects in 3D space in an image.

Parameters:
 - object_type: Very generally, the type of object you're detecting (e.g. headphones, book, cup, block, etc.)
 - object_description: More detailed description of the object you would like to detect (e.g. "red book", "left cup", etc.)
 
Returns:
 - object: The selected `Object` that has an (x, y, z) position.

### `Robot` class

Methods:

1. `robot.move_to([x, y, z])`: Moves the robot arm's hand to a specified location.
Parameters:
 - [x, y, z]: The 3D position for the robot's gripper to move to.
 
2. `robot.move_by([dx, dy, dz])`: Moves the robot arm's hand by a specific relative amount (in meters).

3. `robot.grasp(object)`: Grasps an object.
Parameters: 
 - `object`, a variable of type `Object` (which is an item of the list returned by `scene.detect`)

4. `robot.release()`: Releases the grasped object.

### `Human` class

#### `human.ask(question)` function
If you have no existing instructions, or you have uncertainty about what you should do next, your code
block should consist of a call to the `ask()` function. This will signal to the system that you are
requesting additional instructions or clarification.

Example 1:
```python
human.ask("Which object should I move?")
```

Example 2:
```python
human.ask("I am ready for your next instruction. What would you like me to do?")
```

#### `human.request_object_selection(prompt)` function

If you get errors where no objects are found, but you can see the target object in the scene, call
this function. The object detector in the `scene` class can be buggy. By calling this function, a
human will provide a ground truth object for you to use.

# Examples

## Example 1

(Image placeholder)
Description of image: There are two green cups and a variety of other objects on the table.

Instructions: Stack the cups. It does not matter to me which cup is stacked on which other.

Reasoning:
It seems that the human wants to stack the cups. Because they did not indicate a preference
for the stacked cup, I will choose one arbitrarily. However, in other scenarios, I may want
to be more cautious. In order to stack two objects, I must move one of the objects on top of the other.
There are only two cups in the scene, and so this concludes my general analysis of the intent.

Short plan:
1. Locate the cups.
2. Choose a cup to stack onto the other cup.
3. Grasp the cup we wish to stack.
4. Move the robot's arm up, center it above the other cup, and place the cup we are holding

Code implementation:
```python
# Locate the cups.
target_cup = scene.choose('cup', 'bottom cup for stack')
stack_up = scene.choose('cup', 'cup to place on bottom cup')

# Grasp the cup we wish to stack.
robot.grasp(stack_cup)

# Move the robot's arm up,
robot.move_by([0, 0, 0.3])

# center it above the other cup, and
robot.move_to([base_cup.centroid[0], base_cup.centroid[1], base_cup.centroid[2] + 0.2])

# place the cup we are holding
robot.release()
```

# Your Task

Describe the image you see. Then, write a short plan for how to execute your instructions: {INSTRUCTIONS}
Finally, write a code snippet that uses the provided APIs to control the robot to move to the desired location.
Your answer should include three sections:
- "Reasoning" with a brief consideration of the overall task goals,
- "Short Plan" with a brief natural language description of the plan,
- "Code Implementation", with a Python code block to control the robot in the desired way. Begin the code block with the string "```python"
and end the code block with the string "```".

Assume that `scene` and `robot` are local variables. Additionally, NumPy is available as `np`. However, do
not import or use any other libraries in your code. Assume the robot is running in simulation, and that you
DO have the capability to write code for this scene.

Guidelines:
 * Do not assume you know the location of an object. You must use the `scene.choose` method to locate objects.
 * Perform calls to `scene.choose` BEFORE moving the robot arm.
