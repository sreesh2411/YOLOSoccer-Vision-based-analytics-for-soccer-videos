from utils import read_video, save_video
#, compress_video
from trackers import Tracker
import cv2
import numpy as np
import time
import os
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import subprocess


def main():
    # Read Video
    video_path = 'input_videos/0bfacc_3.mp4'
    video_frames = read_video(video_path)


    # Extract the base name of the video file
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Construct the stub paths based on the input video name
    track_stub_path = f'stubs/track_stubs_{video_base_name}.pkl'
    camera_movement_stub_path = f'stubs/camera_movement_stub_{video_base_name}.pkl'

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=track_stub_path)
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path=camera_movement_stub_path)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)


    # Construct output video filename based on input video filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_name = f'output_video_{base_name}.mp4'
    output_video_path = os.path.join('output_videos', output_video_name)


    # Save video
    save_video(output_video_frames, output_video_path)

    # Construct compressed output video path
    compressed_output_video_path = os.path.join('output_videos', f'output_video_{base_name}_com.mp4')


    ffmpeg_cmd=f'ffmpeg -i {output_video_path} -c:v libx264 -c:a copy -crf 20 {compressed_output_video_path}'
    subprocess.run(ffmpeg_cmd, shell=True)

    os.remove("output_videos/output_video.mp4")
    os.remove("output_videos/output_video_com.mp4")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")