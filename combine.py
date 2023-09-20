from PIL import Image, ImageDraw, ImageFont
import os

# Directory where your images are stored
image_dir = 'C:/Users/dzung/project/people-flow-modified/People-Flows-Modified/plot/24'
original_dir ='C:/Users/dzung/project/people-flow-modified/our_dataset/test_data/24/'
# List of image filenames (assuming they are in the same directory)
image_files = [
    os.path.join(original_dir, '034.jpg'),
    os.path.join(image_dir, '034_24_gt.jpg'),
    os.path.join(image_dir, '034_24_pred.jpg'),
    os.path.join(image_dir, '034_24_flow_1.jpg'),
    os.path.join(image_dir, '034_24_flow_2.jpg'),
    os.path.join(image_dir, '034_24_flow_3.jpg'),
    os.path.join(image_dir, '034_24_flow_4.jpg'),
    os.path.join(image_dir, '034_24_flow_5.jpg'),
    os.path.join(image_dir, '034_24_flow_6.jpg'),
    os.path.join(image_dir, '034_24_flow_7.jpg'),
    os.path.join(image_dir, '034_24_flow_8.jpg'),
    os.path.join(image_dir, '034_24_flow_9.jpg'),
]

# Number of rows and columns in the grid
rows = 4
cols = 3

# Gap size between images (in pixels)
gap = 35

# Size of each image in the grid (assuming all images have the same dimensions)
image_size = (640, 360)  # Replace with your actual image size (width, height)

# Calculate the size of the output image
output_width = cols * (image_size[0] + gap) - gap
output_height = rows * (image_size[1] + gap) - gap +50

# Create a blank output image
output_image = Image.new('RGB', (output_width, output_height), (255, 255, 255))

# Create a drawing context
draw = ImageDraw.Draw(output_image)

# Define the font for labels
font = ImageFont.truetype("arial.ttf", 25)  # You may need to specify the path to your font file

# Paste each image into the grid and add labels
for i, image_file in enumerate(image_files):
    row = i // cols
    col = i % cols
    x = col * (image_size[0] + gap)
    y = row * (image_size[1] + gap)
    
    # Open the image file
    img = Image.open(image_file)
    
    # Resize the image to the desired size
    img = img.resize(image_size)
    
    # Paste the resized image onto the output image
    output_image.paste(img, (x, y))
    
    # Add labels (a, b, c, d, e, etc.)
    label = chr(ord('a') + i) + '.'
    label_size = draw.textsize(label, font=font)
    label_x = x + (image_size[0] - label_size[0]) // 2
    label_y = y + image_size[1] + 5  # Adjust the vertical position as needed
    draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

# Save the final image
output_image.save('output_grid_with_labels.jpg')
