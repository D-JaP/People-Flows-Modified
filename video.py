import cv2
import os

# Parameters
# pred = False
# flow = not pred
original = True
hsv = True
for i in range(1,11):
    image_folder = "./plot/24"
    # if (flow == True): # pred
    #     image_folder = "./plot/24"
    #     images = [img for img in os.listdir(image_folder) if ("flow_"+str(i)) in img and img.endswith(".jpg")]
    #     output_video = "./video/flow_output_video.mp4".replace("flow", "flow"+str(i-1))
    # elif (pred == True):
    #     images = [img for img in os.listdir(image_folder) if ("pred") in img and img.endswith(".jpg")]
    #     output_video = "./video/pred_output_video.mp4"
    # if (original== True): # original
    #     image_folder = "C:/Users/dzung/project/people-flow-modified/our_dataset/test_data/24"
    #     images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    #     output_video = "./video/original_output_video.mp4"
    if (hsv == True):
        images = [img for img in os.listdir(image_folder) if ("hsv") in img and img.endswith(".jpg")]
        output_video = "./video/hsv_output_video.mp4"


    fps = 20  # Frames per second

    # Get a list of all image files in the folder

    # Sort the images to ensure they are in the correct order
    images.sort()
    # Get the first image to determine the size of the video frames
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Loop through the images and add them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the video object
    video.release()

    print(f"Video saved as {output_video}")
