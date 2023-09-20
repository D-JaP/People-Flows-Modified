import cv2
import numpy as np
import os 
# Parameters
video_folder = "./video"
output_video = "./video/output_matrix_video.mp4"
rows = 3
cols = 3
padding_color = (0, 0, 0)  # Black padding
fps = 20  # Frames per second

# Get a list of video files in the folder
videos = [video for video in os.listdir(video_folder) if "flow" in video and video.endswith(".mp4")]

# Sort the videos to ensure they are in the desired order
videos.sort()

# Open the first video to get dimensions
first_video_path = os.path.join(video_folder, videos[0])
first_video = cv2.VideoCapture(first_video_path)
frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the combined video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
combined_video = cv2.VideoWriter(output_video, fourcc, fps, (frame_width * cols, frame_height * rows))

# Loop through the videos and create the matrix layout
for i in range(0, len(videos), rows * cols):
    frame_matrix = []
    for j in range(rows * cols):
        if i + j < len(videos):
            video_path = os.path.join(video_folder, videos[i + j])
            video = cv2.VideoCapture(video_path)
            ret, frame = video.read()
            if ret:
                frame = cv2.resize(frame, (frame_width, frame_height))  # Resize to match the frame dimensions
                frame_matrix.append(frame)
            video.release()

    if len(frame_matrix) == rows * cols:
        combined_frame = cv2.vconcat([cv2.hconcat(frame_matrix[:cols]), cv2.hconcat(frame_matrix[cols:2*cols]), cv2.hconcat(frame_matrix[2*cols:])])
        combined_video.write(combined_frame)

combined_video.release()

print(f"Matrix video saved as {output_video}")
