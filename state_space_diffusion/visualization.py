import torch
import matplotlib.pyplot as plt
import numpy as np

from demo_to_state_action_pairs import create_labels_v3
import quaternions as Q

def draw_rotation_matrix(position, rotation_matrix):
    for (axis, color) in zip(rotation_matrix, 'rgb'):
        plt.quiver(position[0], position[1], axis[0], axis[1], scale=10, color=color, alpha=0.5)


NP = lambda x: x.detach().cpu().numpy()

def visualize_2d_score_prediction(image, sample_grid, predicted_direction_px, true_direction_px):
    if type(image) == torch.Tensor:
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
    
    plt.imshow(image, origin="lower")
    plt.quiver(
        NP(sample_grid[:, 0]),
        NP(sample_grid[:, 1]),
        NP(predicted_direction_px[:, 0]),
        NP(predicted_direction_px[:, 1]),
        color='red',
        label='Predicted'
    )
    plt.quiver(
        NP(sample_grid[:, 0]),
        NP(sample_grid[:, 1]),
        NP(true_direction_px[:, 0]),
        NP(true_direction_px[:, 1]),
        color='blue',
        label='True'
    )

def visualize_2d_sampling_trajectory(
    images,
    score_map_coordinates,
    predicted_score_maps,
    true_score_maps,
    history_2d,
    target_locations_2d,
    camera_names):
    # Plotting the score function in 2D.
    for i, (name, image, trajectory_2d, predicted_score_map, true_score_map, target_location_2d) in enumerate(zip(
        camera_names,
        images,
        history_2d.permute(1, 0, 2),
        predicted_score_maps,
        true_score_maps,
        target_locations_2d.detach().cpu().numpy(),
    )):
        # convert image to numpy
        if type(image) == torch.Tensor:
            image = image.detach().cpu().numpy().transpose(1, 2, 0)
        
        plt.subplot(1, len(images), i + 1)
        plt.title(name + " view")
        plt.imshow(image)
        # plt.xlim(0, 224)
        # plt.ylim(224, 0)
        
        plt.scatter(
            target_location_2d[0],
            target_location_2d[1],
            c=[(1.0, 0.5, 0.0)],
            label='Target position'
        )
        
        # flip score function because of the flipped y axis (results from the ylim stuff, not a
        # matrix projection bug i think)
        # predicted_score_map[..., 1] *= -1
        # true_score_map[..., 1] *= -1
        
        # Visualize the sampling trajectory.
        trajectory_2d = trajectory_2d.cpu().numpy()
        plt.scatter(
            trajectory_2d[..., 0],
            trajectory_2d[..., 1],
            label="Sampling trajectory",
            c=np.arange(len(trajectory_2d)),
            cmap="viridis",
        )
        
        # Render the score function as a vector field.
        smc = score_map_coordinates.view(-1, 2)
        psm = predicted_score_map.view(-1, 2)
        tsm = true_score_map.view(-1, 2)
        smc = NP(smc)
        psm = NP(psm)
        tsm = NP(tsm)
        arrow_scale = 0.1
        
        use_quiver = True
        if use_quiver:
            plt.quiver(
                smc[:, 0],
                smc[:, 1],
                psm[:, 0] * arrow_scale,
                psm[:, 1] * arrow_scale * -1, # fix bug during imshow
                # scale=1,
                color='r',
                label='Predicted Score'
            )
            plt.quiver(
                smc[:, 0],
                smc[:, 1],
                tsm[:, 0] * arrow_scale,
                tsm[:, 1] * arrow_scale * -1, # fix bug during imshow
                # scale=1,
                color='b',
                label='True Score'
            )
        # else:
            # plt.quiver(
            #     smc[:, 0],
            #     smc[:, 1],
            #     psm[:, 0] * arrow_scale,
            #     psm[:, 1] * arrow_scale * -1, # fix bug during imshow
            #     # scale=1,
            #     color='r',
            #     label='Predicted Score'
            # )
            # plt.quiver(
            #     smc[:, 0],
            #     smc[:, 1],
            #     tsm[:, 0] * arrow_scale,
            #     tsm[:, 1] * arrow_scale * -1, # fix bug during imshow
            #     # scale=1,
            #     color='b',
            #     label='True Score'
            # )
        
        # Visualize the quaternion
        # show_quaternion = False
        # if show_quaternion:
        #     for j in range(1, len(history_quats)):
        #         # Take true quaternion and invert camera rotation.
        #         rotation_matrix = extrinsics[i][:3, :3].T @ Q.quaternion_to_rotation_matrix(history_quats[j])
        #         draw_rotation_matrix(trajectory_2d[j - 1], rotation_matrix)
            
        # plt.legend()

def evaluate(
    demo,
    keypoint_index,
    output_prefix,
    starting_point,
    score_maps_to_visualize,
    chosen_cameras,
):
    state_action_tuples = create_labels_v3(demo)

    # Test with first keypoint.
    images, positions, quaternions, info = state_action_tuples[keypoint_index]
    extrinsics = info['extrinsics']
    intrinsics = info['intrinsics']
    
    # Select the subset of cameras to use.
    

    # Prepare image/position
    images = [prepare_image(image) for image in images]
    positions = torch.tensor(positions) * 224/128
    
    history, history_quats, history_2d, score_maps, quat_maps = sample(
        model,
        images,
        extrinsics,
        intrinsics,
        starting_point,
        chosen_cameras
    )
    
    grid_size = 14

    ### Plot 2D. ###
    plt.clf()
    plt.figure(figsize=(6 * len(images), 6))
    visualize_2d_sampling_trajectory(images, score_maps, history_2d, chosen_cameras)
    plt.tight_layout()
    plt.savefig(output_prefix + "_2d_multiview_sampling_trajectory.png", dpi=256)
    plt.clf()

    ### Plot 3D. ###
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    buffer = 0.2
    ax.set_xlim(-0.3 - buffer, 0.7 + buffer)
    ax.set_ylim(-0.5 - buffer, 0.5 + buffer)
    ax.set_zlim(0.6 - buffer, 1.6 + buffer)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot(*history.cpu().numpy().T, c='r')
    
    # get depth data
    start_obs = info['start_obs']
    point_cloud_camera = chosen_cameras[0]
    point_cloud = getattr(start_obs, point_cloud_camera + '_point_cloud')
    points = point_cloud.reshape(-1, 3)
    colors = getattr(start_obs, point_cloud_camera + '_rgb')
    colors = colors.reshape(-1, 3) / 255.0
    ax.scatter(*points.T, c=colors, s=2, alpha=0.1)
    
    ax.set_box_aspect([1, 1, 1])

    true_pos = info['target_obs'].gripper_pose[:3]
    ax.scatter([true_pos[0]], [true_pos[1]], [true_pos[2]], color='r', label="True Position")
    start_pos = history[0].cpu().numpy()
    ax.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]], color='g', label="Start Position")
    end_pos = history[-1].cpu().numpy()
    ax.scatter([end_pos[0]], [end_pos[1]], [end_pos[2]], color='b', label="End Position")
    
    print(start_pos, end_pos)
    
    # plot the extrinsics for each camera
    for i, extrinsic in enumerate(extrinsics):
        t = extrinsic[:3, 3]
        r = extrinsic[:3, :3].T
        scale = 0.5
        if i in chosen_cameras:
            for ax_, c in [(r[0], 'r'), (r[1], 'g'), (r[2], 'b')]:
                ax.plot([
                    t[0].item(), (t[0] + ax_[0] * scale).item()
                ], [
                    t[1].item(), (t[1] + ax_[1] * scale).item()
                ], [
                    t[2].item(), (t[2] + ax_[2] * scale).item()
                ], color=c)
                
                if c == 'b':
                    print("true z:", ax_)

    if len(score_maps_to_visualize) > 0:
        visualize_3d_score_function(
            [score_maps[i] for i in score_maps_to_visualize],
            [extrinsics[i] for i in score_maps_to_visualize],
            [intrinsics[i] for i in score_maps_to_visualize],
            ax=ax
        )
        
    plt.legend()
    plt.show()
    # plt.savefig(output_prefix + "_3d_sampling_trajectory.png")
    # plt.close()
