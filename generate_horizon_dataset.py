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
import numpy as np
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt

def smoothen_with_spline(timestamps: np.ndarray, values: np.ndarray, sample_interval=0.02, show=False):
    # Generate some example data (time and corresponding values)
    np.random.seed(0)

    # Interpolate the data using a spline
    times = timestamps - timestamps[0]
    tck_s = splrep(times, values, s=0.0005)
    f = BSpline(*tck_s)

    # Generate new time points for interpolation
    sampled_times = np.arange(0, timestamps[-1] - timestamps[0] + sample_interval, sample_interval)

    # Interpolate the values at the new time points
    sampled_values = f(sampled_times)

    if show:
        # Plot the original and interpolated data
        plt.figure(figsize=(10, 6))
        plt.plot(times, values, 'o', label='Original data')
        plt.plot(sampled_times, sampled_values, '-', label='Interpolated data (spline)')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Temporal interpolation with a spline')
        plt.show()

    return (sampled_times, sampled_values)

def main():
    with open("recordings/recording_011_drawer_play_data_high_sample_rate/annotated/annotated_recording.pkl", "rb") as f:
        recording = pickle.load(f)

    visualize = False
    dataset = []

    if visualize:
        visualizer = ObjectDetectionVisualizer()
    else:
        visualizer = None

    horizon_frames = 40
    speed = 2
    prev_timestamp = recording[0]['timestamp']
    for i in range(1, len(recording)):
        step = recording[i]
        # Simulate realistic playback speed
        if visualize:
            time.sleep((step['timestamp'] - prev_timestamp) / speed)
        prev_timestamp = step['timestamp']

        times = []
        X = []
        Y = []
        Z = []

        for j in range(i, min(i + horizon_frames, len(recording))):
            pos = recording[j]['hand_position']['3d']
            if pos is not None:
                x, y, z = pos
                times.append(recording[j]['timestamp'])
                X.append(x)
                Y.append(y)
                Z.append(z)

        if len(times) < 0.8 * horizon_frames:
            continue

        times = np.array(times)
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        times_smooth, X_smooth = smoothen_with_spline(times, X)
        _, Y_smooth = smoothen_with_spline(times, Y)
        _, Z_smooth = smoothen_with_spline(times, Z)

        dataset.append((times_smooth, X_smooth, Y_smooth, Z_smooth))

        if visualize:
            visualizer.show(([
                ('Current', step['hand_position']['3d'], 5, (0.0, 1.0, 0.0)),
            ] if step['hand_position']['3d'] is not None else [])
            + [
                ('Target', (X, Y, Z), 2, (1.0, 1.0, 1.0)),
            ])

    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    main()
