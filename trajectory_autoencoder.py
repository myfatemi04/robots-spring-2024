"""
Note that because we only consider a certain horizon of each trajectory,
we could actually just record one long video and make crops out of that.

Additionally, we could use a video captioning agent to describe each capture
and format the captions as instructions.

We could even take a Whisper model and use it to caption the audio, and then
we can label our data just by narrating along, saying things like "And now,
I will open the top drawer."

Anyway. For now, let's just inspect the data.
"""

import pickle
import time

from visualization import ObjectDetectionVisualizer

with open("recordings/recording_010_open_drawers_long/recording.pkl", "rb") as f:
    recording = pickle.load(f)

visualizer = ObjectDetectionVisualizer()
speed = 2
prev_timestamp = recording[0]['timestamp']
for i in range(1, len(recording)):
    step = recording[i]
    # Simulate realistic playback
    time.sleep((step['timestamp'] - prev_timestamp) / speed)

    X = []
    Y = []
    Z = []
    for j in range(i, min(i + 10, len(recording))):
        pos = recording[j]['hand_position']['3d']
        if pos is not None:
            x, y, z = pos
            X.append(x)
            Y.append(y)
            Z.append(z)

    visualizer.show(([
        ('Current', step['hand_position']['3d'], 5, (0.0, 1.0, 0.0)),
    ] if step['hand_position']['3d'] is not None else [])
    + [
        ('Target', (X, Y, Z), 5, (1.0, 1.0, 1.0)),
    ])

    prev_timestamp = step['timestamp']
